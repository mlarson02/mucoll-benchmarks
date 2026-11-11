// Quick check of the stau digi (e.g. digi_fullBIB_coned_loose_timing_rawtime):
//  - every event has staus: |PDG| 1000015 / 2000015, skipping zero-length copies
//    (endpoint = vertex), as LLPStudies/MarkLLPCode/analysis/analysis.py does
//  - the coned tracker hits contain the staus' hits:
//      stau sim hits    hits in the full sim collection whose MCParticle is a stau
//      kept in cone     of those, the ones in the coned sim subset (<sim>Coned)
//      in-cone hits     coned digitised hits within coneMm of a stau line; the
//                       staus have pT ~ TeV, so a straight line is within < 1 mm
//                       of the helix over the tracker
//  - stau hit delay t - |x|/c of the coned digitised hits (beta ~ 0.6 -> late)
// Run inside the mucoll container (podio + edm4hep libraries):
//   root -l -b -q 'verifyStauDigi.C("/ospool/uc-shared/project/futurecolliders/larsonma/stau_4000GeV_30ns/digi_fullBIB_coned_loose_timing_rawtime", 3)'
// Each digi file is ~3.4 GB (2 events), so keep nFiles small.

R__LOAD_LIBRARY(libpodioRootIO)
R__LOAD_LIBRARY(libedm4hep)
R__LOAD_LIBRARY(libedm4hepDict)

#include "podio/Frame.h"
#include "podio/ROOTReader.h"
#include "edm4hep/MCParticleCollection.h"
#include "edm4hep/SimTrackerHitCollection.h"
#include "edm4hep/TrackerHitPlaneCollection.h"

#include "TString.h"
#include "TSystem.h"
#include "TSystemDirectory.h"
#include "TList.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

namespace {

const double kC = 299.792458;  // mm/ns

bool isStau(int pdg) { return std::abs(pdg) == 1000015 || std::abs(pdg) == 2000015; }

struct Line { double vx, vy, vz, ux, uy, uz; };

struct SubDet {
  std::string prefix, sim;  // digitiser prefix, base sim-hit collection
  long nConed = 0, nInCone = 0, nStauSim = 0, nStauConed = 0;
  double sumDelay = 0, maxDelay = -1e9;
};

// distance of (x, y, z) to the line, forward of its vertex only (-1 behind it)
double lineDist(const Line& l, double x, double y, double z) {
  const double dx = x - l.vx, dy = y - l.vy, dz = z - l.vz;
  const double t = dx * l.ux + dy * l.uy + dz * l.uz;
  if (t < 0) return -1;
  const double px = dx - t * l.ux, py = dy - t * l.uy, pz = dz - t * l.uz;
  return std::sqrt(px * px + py * py + pz * pz);
}

// digi files in dir, by chunk id
std::vector<std::pair<int, TString>> digiFiles(const TString& dir) {
  std::vector<std::pair<int, TString>> out;
  TSystemDirectory d("d", dir);
  TList* files = d.GetListOfFiles();
  if (!files) return out;
  for (TObject* o : *files) {
    TString n = o->GetName();
    const int i = n.Index("_digi");
    if (i < 0 || !n.EndsWith(".edm4hep.root")) continue;
    TString id = n(i + 5, n.Length() - (i + 5) - 13);
    if (!id.IsDigit()) continue;
    out.push_back({id.Atoi(), dir + "/" + n});
  }
  std::sort(out.begin(), out.end());
  return out;
}

}  // namespace

void verifyStauDigi(TString dir = "/ospool/uc-shared/project/futurecolliders/larsonma/stau_4000GeV_30ns/digi_fullBIB_coned_loose_timing_rawtime",
                    int nFiles = 3, double coneMm = 30.) {
  std::vector<SubDet> dets = {
    {"VXDBarrel", "VertexBarrelCollection"},       {"VXDEndcap", "VertexEndcapCollection"},
    {"ITBarrel", "InnerTrackerBarrelCollection"},  {"ITEndcap", "InnerTrackerEndcapCollection"},
    {"OTBarrel", "OuterTrackerBarrelCollection"},  {"OTEndcap", "OuterTrackerEndcapCollection"},
  };

  auto files = digiFiles(dir);
  printf("%zu digi files in %s, reading the first %d\n", files.size(), dir.Data(), nFiles);
  if ((int)files.size() > nFiles) files.resize(nFiles);

  long nEvents = 0, nNoStau = 0, nNoStauHits = 0, nStaus = 0;
  std::map<int, long> stauStatus;  // generator status of every stau entry
  std::vector<std::string> missing;

  for (auto& [chunk, path] : files) {
    podio::ROOTReader reader;
    reader.openFile(path.Data());
    const unsigned nEnt = reader.getEntries("events");
    for (unsigned ev = 0; ev < nEnt; ++ev) {
      const podio::Frame frame(reader.readNextEntry("events"));
      const auto avail = frame.getAvailableCollections();
      auto has = [&](const std::string& n) {
        if (std::find(avail.begin(), avail.end(), n) != avail.end()) return true;
        if (std::find(missing.begin(), missing.end(), n) == missing.end()) missing.push_back(n);
        return false;
      };
      ++nEvents;

      // staus
      std::vector<Line> staus;
      TString stauInfo;
      if (has("MCParticles")) {
        for (const auto& p : frame.get<edm4hep::MCParticleCollection>("MCParticles")) {
          if (!isStau(p.getPDG())) continue;
          ++stauStatus[p.getGeneratorStatus()];
          const auto v = p.getVertex();
          const auto e = p.getEndpoint();
          const auto m = p.getMomentum();
          const double len = std::sqrt(std::pow(e.x - v.x, 2) + std::pow(e.y - v.y, 2) + std::pow(e.z - v.z, 2));
          if (len == 0) continue;
          const double pp = std::sqrt(m.x * m.x + m.y * m.y + m.z * m.z);
          const double beta = pp / std::sqrt(pp * pp + p.getMass() * p.getMass());
          staus.push_back({v.x, v.y, v.z, m.x / pp, m.y / pp, m.z / pp});
          stauInfo += Form(" [pdg %d st %d q %+.0f pT %.0f beta %.2f end r %.0f z %.0f]", p.getPDG(),
                           p.getGeneratorStatus(), p.getCharge(), std::hypot(m.x, m.y), beta,
                           std::hypot(e.x, e.y), e.z);
        }
      }
      nStaus += staus.size();
      if (staus.empty()) ++nNoStau;

      // hits
      long evStauConed = 0;
      TString hitInfo;
      for (auto& d : dets) {
        const std::string hName = d.prefix + "HitsConed", sName = d.sim + "Coned";
        long stauConed = 0;
        if (has(hName) && has(sName)) {
          const auto& hits = frame.get<edm4hep::TrackerHitPlaneCollection>(hName);
          const auto& sims = frame.get<edm4hep::SimTrackerHitCollection>(sName);
          d.nConed += hits.size();
          for (const auto& h : hits) {
            const auto x = h.getPosition();
            for (const auto& l : staus) {
              const double dist = lineDist(l, x.x, x.y, x.z);
              if (dist >= 0 && dist < coneMm) { ++d.nInCone; break; }
            }
          }
          // the coner writes the coned hits and sim hits in the same order
          for (size_t i = 0; i < sims.size(); ++i) {
            const auto part = sims[i].getParticle();
            if (!part.isAvailable() || !isStau(part.getPDG())) continue;
            ++stauConed;
            if (i < hits.size()) {
              const auto x = hits[i].getPosition();
              const double delay = hits[i].getTime() - std::sqrt(x.x * x.x + x.y * x.y + x.z * x.z) / kC;
              d.sumDelay += delay;
              d.maxDelay = std::max(d.maxDelay, delay);
            }
          }
        }
        if (has(d.sim)) {
          for (const auto& s : frame.get<edm4hep::SimTrackerHitCollection>(d.sim)) {
            const auto part = s.getParticle();
            if (part.isAvailable() && isStau(part.getPDG())) ++d.nStauSim;
          }
        }
        d.nStauConed += stauConed;
        evStauConed += stauConed;
        hitInfo += Form(" %s %ld", d.prefix.c_str(), stauConed);
      }
      if (!staus.empty() && evStauConed == 0) ++nNoStauHits;

      printf("chunk %d ev %u: %zu staus%s\n    coned stau hits:%s%s\n", chunk, ev, staus.size(), stauInfo.Data(),
             hitInfo.Data(), staus.empty() ? "   <-- NO STAU" : (evStauConed == 0 ? "   <-- NO CONED STAU HITS" : ""));
    }
  }

  printf("\n==== %ld events, %ld moving staus (%.2f / event)\n", nEvents, nStaus, nEvents ? double(nStaus) / nEvents : 0.);
  printf("stau entries by generator status:");
  for (auto& [st, n] : stauStatus) printf("  %d: %ld", st, n);
  printf("\nevents without a stau: %ld, with staus but no coned stau hits: %ld\n", nNoStau, nNoStauHits);
  if (!missing.empty()) {
    printf("collections missing in some events:");
    for (auto& n : missing) printf(" %s", n.c_str());
    printf("\n");
  }
  printf("\n%-10s %12s %10s %12s %12s %9s %12s %12s\n", "subdet", "coned/ev", "in cone", "stauSim/ev", "stauConed/ev",
         "kept", "delay mean", "delay max");
  for (auto& d : dets) {
    const double ne = std::max<long>(nEvents, 1);
    printf("%-10s %12.1f %9.1f%% %12.1f %12.1f %8.1f%% %10.3fns %10.3fns\n", d.prefix.c_str(), d.nConed / ne,
           d.nConed ? 100. * d.nInCone / d.nConed : 0., d.nStauSim / ne, d.nStauConed / ne,
           d.nStauSim ? 100. * d.nStauConed / d.nStauSim : 0., d.nStauConed ? d.sumDelay / d.nStauConed : 0.,
           d.nStauConed ? d.maxDelay : 0.);
  }
  printf("\nin cone: coned hits within %.0f mm of a stau line. kept: stau sim hits that made it into the coned\n"
         "subset (loses those outside the digi time window too). delay: coned stau hits' time - |x|/c.\n", coneMm);
}
