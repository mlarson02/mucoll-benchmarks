// trackPlotter.C -- stage 2 of the track analysis (stage 1: trackNtupler.py).
//
// Reads the ntuple directories written by the condor/ntuple jobs, one
// directory per input, with a TChain, and makes the per-input plots and
// printouts and the cross-input overlays of edm4hepTrackAnalyzer.py.
// Inputs, legend labels and the overlay directory are set in trackPlotter()
// at the bottom of this file.
//
//   root -l -b -q trackPlotter.C+
//   root -l -b -q 'trackPlotter.C+("<ntuple dir>")'   one directory only, no overlays (quick test)
//
// Per-input plots go to ./<ntuple directory name>_plots/, overlays to the
// overlay directory.

#include "TROOT.h"
#include "TStyle.h"
#include "TSystem.h"
#include "TError.h"
#include "TChain.h"
#include "TTreeReader.h"
#include "TTreeReaderValue.h"
#include "TTreeReaderArray.h"
#include "TCanvas.h"
#include "TPad.h"
#include "TH1D.h"
#include "TH1F.h"
#include "TH2D.h"
#include "TGraph.h"
#include "TGraphErrors.h"
#include "TLegend.h"
#include "TLine.h"
#include "TEllipse.h"
#include "TLatex.h"
#include "TColor.h"
#include "TString.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace tp {

using Vec = std::vector<double>;
const double NaN = std::numeric_limits<double>::quiet_NaN();

// ---- settings ----------------------------------------------------------------
int MAX_FILES_PER_INPUT = -1;               // > 0: read only the first N ntuple files of each input
const bool ETA_ZERO = true;                 // eta~0 tracks: zoom the r-z maps to z in [-500, 500] mm
const double RZ_ZLIM = ETA_ZERO ? 500 : 2100;
const double ZSEAM_ZOOM_HALF = 1.0;         // x-range of the zoomed z = 0 scatter [mm]
const Long64_t ZSEAM_MAX_HITS = 2000000;    // cap on the hits near z = 0 kept per input
const int ID_DIAG_TARGET_HITS_PER_SYSTEM = 5000;
const int OCC_NZ = 460, OCC_NR = 160;       // occupancy map: 10 mm bins
const double OCC_ZMIN = -2300, OCC_ZMAX = 2300, OCC_RMAX = 1600;
const double FILT_MIN_PT = 0.5, FILT_MAX_D0 = 100, FILT_MAX_Z0 = 100;   // AllTracks -> SiTracks filter
const int PRF_NHITS_MIN = 8;

const char* SYS_REL[7] = {"?", "VXDBarrel", "VXDEndcap", "ITBarrel", "ITEndcap", "OTBarrel", "OTEndcap"};
const char* SYS_NAMES[7] = {"?", "VXD Barrel", "VXD Endcap", "IT Barrel", "IT Endcap", "OT Barrel", "OT Endcap"};
const char* DIAG_NAMES[6] = {"VXDBarrelHits", "VXDEndcapHits", "ITBarrelHits",
                             "ITEndcapHits", "OTBarrelHits", "OTEndcapHits"};
const char* relname(int s) { return (s >= 1 && s <= 6) ? SYS_REL[s] : "?"; }

// Tracker outlines for the event displays [mm].
const double vb_r[4] = {30, 51, 74, 102};
const double vb_l = 130;
const double ve_z[4] = {80, 120, 200, 280};
const double ve_h[4] = {0, 15, 35, 50};
const double ib_r[3] = {127, 340, 554};
const double ib_l[3] = {963.2, 963.2, 1384.6};
const double ie_z[7] = {524, 808, 1093, 1377, 1661, 1946, 2190};
const double ob_r[3] = {819, 1153, 1486};
const double ob_l = 2528.4;
const double oe_z[4] = {1310, 1617, 1883, 2190};
double ie_h(int j) { return 100 + j * (300. - 100.) / 7; }

// matplotlib's colour cycle and the named colours the old plots used.
const char* MPL[10] = {"#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                       "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"};
int mpl(int i) { return TColor::GetColor(MPL[i % 10]); }
int steelblue() { return TColor::GetColor("#4682b4"); }
int salmon() { return TColor::GetColor("#fa8072"); }
int skyblue() { return TColor::GetColor("#87ceeb"); }
int lightgreen() { return TColor::GetColor("#90ee90"); }
int green() { return TColor::GetColor("#008000"); }
int purple() { return TColor::GetColor("#800080"); }
int orange() { return TColor::GetColor("#ff7f0e"); }
int gray() { return TColor::GetColor("#808080"); }

// ---- data of one input ---------------------------------------------------------
struct IdSample {
  double u;
  float r, z;
  int side, layer;
};

// One event's hit maps: kind 0 = muon hits on the matched track, 1 = its other
// hits, 2 = muon hits it missed.
struct Display {
  Long64_t event = 0;
  Vec x[3], y[3], z[3], r[3];
  double chi2Mean = NaN, truthPt = NaN, recoPt = NaN;
};

struct InputData {
  TString dir, tag, label;
  int nFiles = 0;
  Long64_t nEvents = 0;
  bool haveMode = false, noHits = false, detectedNoHits = false, nhitsUnavailable = false;
  bool useConed = false, mixedModes = false;
  int signalPdg = 13, seedMinHits = 3;
  double seedMaxDr = 0.05, zseamCollectHalf = 25.0;
  // per event
  Vec allMult, dedupMult, siMult, seedMult, muonMatchedSi;
  std::vector<std::map<int, int>> layerNhits;   // (system * 64 + layer) -> hits, events with hit collections
  // per signal muon
  Vec effEta, effPt, effPhi, effTheta, effSi, effAll;
  long matchedNonrefit = 0, matchedAllTracks = 0;
  // muon-matched SiTracks
  Vec chi2Nr, etaNr, truthPtNr, recoPtNr, ptResNr, phiNr, d0Nr, z0Nr, d0ResNr, z0ResNr, dqop2Nr;
  Vec muFracNr, nbibNr, pixNr, innerNr, outerNr, totalNr;
  std::map<int, std::map<int, long>> layerHitsNr;   // system -> layer -> hits
  // muon-matched AllTracks
  Vec chi2All, pixAll, innerAll, outerAll, totalAll;
  // every AllTracks track / seed
  std::vector<float> rawNhits, rawChi2, rawPt, rawD0, rawZ0, seedPt, seedEta;
  std::vector<std::pair<double, double>> seedBest;   // (truth pT, seed pT)
  long nSeedHit = 0, nSeedDr = 0;
  // hits near z = 0
  std::vector<float> zsZ, zsR;
  std::vector<int> zsSys;
  std::vector<Display> displays, allRz;
  std::vector<TString> diagLines;
  // per-file summaries, summed
  Vec occ;
  std::map<std::array<int, 3>, Long64_t> idCounts;   // (system, side, layer) -> hits
  std::map<int, Long64_t> idSeen;                    // system -> hits
  std::map<int, std::vector<IdSample>> idSample;     // system -> random hit sample
};

// ---- numerics --------------------------------------------------------------------
template <class T> double vmean(const std::vector<T>& v) {
  if (v.empty()) return NaN;
  double s = 0;
  for (auto x : v) s += x;
  return s / v.size();
}
template <class T> double vstd(const std::vector<T>& v) {   // population, like np.std
  if (v.empty()) return NaN;
  double m = vmean(v), s = 0;
  for (auto x : v) s += (x - m) * (x - m);
  return std::sqrt(s / v.size());
}
template <class T> double vmedian(std::vector<T> v) {   // like np.median
  if (v.empty()) return NaN;
  size_t n = v.size();
  std::nth_element(v.begin(), v.begin() + n / 2, v.end());
  double hi = v[n / 2];
  if (n % 2) return hi;
  double lo = *std::max_element(v.begin(), v.begin() + n / 2);
  return 0.5 * (lo + hi);
}
template <class T> double vmin(const std::vector<T>& v) { return v.empty() ? NaN : *std::min_element(v.begin(), v.end()); }
template <class T> double vmax(const std::vector<T>& v) { return v.empty() ? NaN : *std::max_element(v.begin(), v.end()); }
template <class T> Vec toVec(const std::vector<T>& v) { return Vec(v.begin(), v.end()); }

Vec linspace(double a, double b, int n) {
  Vec e(n);
  for (int i = 0; i < n; ++i) e[i] = (n == 1) ? a : a + (b - a) * i / (n - 1);
  return e;
}
// n points from lo to hi, evenly spaced in log (np.logspace(log10(lo), log10(hi), n)).
Vec logspace(double lo, double hi, int n) {
  Vec e = linspace(std::log10(lo), std::log10(hi), n);
  for (auto& x : e) x = std::pow(10., x);
  return e;
}
Vec arange(double a, double b, double step) {   // like np.arange
  Vec e;
  int n = (int)std::ceil((b - a) / step - 1e-9);
  for (int i = 0; i < n; ++i) e.push_back(a + i * step);
  return e;
}

// ---- drawing helpers -----------------------------------------------------------------
int gId = 0;
TString uid(const char* prefix = "h") { return TString::Format("%s_%d", prefix, gId++); }

TH1D* H1(const Vec& edges, const TString& title = "") {
  return new TH1D(uid(), title, (int)edges.size() - 1, edges.data());
}

// numpy puts a value equal to the last edge in the last bin; ROOT in the overflow.
double inLast(const TAxis* a, double x) { return x == a->GetXmax() ? a->GetBinCenter(a->GetNbins()) : x; }

enum Norm { kCounts, kPerEntry, kUnit };   // raw counts / 1 over all entries / 1 over the bins (ROOT 1/Integral)
template <class T> TH1D* fillH(TH1D* h, const std::vector<T>& v, Norm norm = kCounts) {
  for (auto x : v)
    if (std::isfinite((double)x)) h->Fill(inLast(h->GetXaxis(), x));
  if (norm == kPerEntry && !v.empty()) h->Scale(1.0 / v.size());
  if (norm == kUnit && h->Integral() > 0) h->Scale(1.0 / h->Integral());
  return h;
}
template <class T> TH1D* hist(const std::vector<T>& v, const Vec& edges, Norm norm = kCounts, const TString& title = "") {
  return fillH(H1(edges, title), v, norm);
}

void stepStyle(TH1* h, int col, int ls = 1, int lw = 2) {
  h->SetLineColor(col);
  h->SetLineStyle(ls);
  h->SetLineWidth(lw);
  h->SetFillStyle(0);
}
void barStyle(TH1* h, int col, double alpha = 1., int edge = kBlack) {
  h->SetFillColorAlpha(col, alpha);
  h->SetFillStyle(1001);
  h->SetLineColor(edge);
  h->SetLineWidth(1);
}

// Draws the histograms on one set of axes, with a y range covering all of them.
void drawAll(const std::vector<TH1*>& hs, bool logy, const char* opt = "HIST") {
  if (hs.empty()) return;
  double mx = 0, mn = 1e300;
  for (auto* h : hs)
    for (int b = 1; b <= h->GetNbinsX(); ++b) {
      double c = h->GetBinContent(b);
      if (c > mx) mx = c;
      if (c > 0 && c < mn) mn = c;
    }
  if (mx <= 0) mx = 1;
  if (logy) {
    hs[0]->SetMinimum(mn < 1e300 ? mn / 2 : 0.5);
    hs[0]->SetMaximum(mx * 3);
  } else {
    hs[0]->SetMinimum(0);
    hs[0]->SetMaximum(mx * 1.1);
  }
  gPad->SetLogy(logy);
  for (size_t i = 0; i < hs.size(); ++i) hs[i]->Draw(i ? TString(opt) + " SAME" : TString(opt));
}

TCanvas* canvas(int w = 800, int h = 600) {
  auto* c = new TCanvas(uid("c"), "", w, h);
  c->SetLeftMargin(0.13);
  c->SetRightMargin(0.05);
  c->SetBottomMargin(0.12);
  c->SetTopMargin(0.09);
  return c;
}

// nx x ny pads, numbered row by row from 1, under an optional overall title.
void grid(TCanvas* c, int nx, int ny, const TString& suptitle = "") {
  double top = suptitle.IsNull() ? 1.0 : 0.95;
  c->cd();
  for (int iy = 0; iy < ny; ++iy)
    for (int ix = 0; ix < nx; ++ix) {
      int k = iy * nx + ix + 1;
      auto* p = new TPad(TString::Format("%s_p%d", c->GetName(), k), "", double(ix) / nx,
                         top * (1 - double(iy + 1) / ny), double(ix + 1) / nx, top * (1 - double(iy) / ny));
      p->SetNumber(k);
      p->SetLeftMargin(0.14);
      p->SetRightMargin(0.05);
      p->SetBottomMargin(0.13);
      p->SetTopMargin(0.1);
      p->Draw();
    }
  if (!suptitle.IsNull()) {
    TLatex t;
    t.SetTextAlign(22);
    t.SetTextSize(0.03);
    t.DrawLatexNDC(0.5, 0.975, suptitle);
  }
}

void save(TCanvas* c, const TString& path) {
  c->SaveAs(path);
  delete c;
}

TLegend* legend(double x1 = 0.55, double y1 = 0.72, double x2 = 0.94, double y2 = 0.9, double ts = 0.03) {
  auto* l = new TLegend(x1, y1, x2, y2);
  l->SetBorderSize(0);
  l->SetFillStyle(0);
  l->SetTextSize(ts);
  return l;
}
// One legend entry over several lines: the symbol on the first.
void addLines(TLegend* l, TObject* o, const char* opt, const std::vector<TString>& lines) {
  for (size_t i = 0; i < lines.size(); ++i) l->AddEntry(i ? (TObject*)nullptr : o, lines[i], i ? "" : opt);
}

// Vertical line across the current pad's frame (log axes handled).
void vline(double x, int col, int style = 2, int width = 1, double alpha = 1.) {
  gPad->Update();
  double xu = gPad->GetLogx() ? std::log10(x) : x;
  auto* l = new TLine(xu, gPad->GetUymin(), xu, gPad->GetUymax());
  l->SetLineColorAlpha(col, alpha);
  l->SetLineStyle(style);
  l->SetLineWidth(width);
  l->Draw();
}

void emptyPanel(const TString& title) {
  auto* f = gPad->DrawFrame(0, 0, 1, 1, title);
  f->GetXaxis()->SetLabelSize(0);
  f->GetYaxis()->SetLabelSize(0);
}

TGraph* scatter(const Vec& x, const Vec& y, int col, double alpha, int marker = 20, double size = 0.6) {
  auto* g = new TGraph((int)x.size(), x.data(), y.data());
  g->SetMarkerStyle(marker);
  g->SetMarkerSize(size);
  g->SetMarkerColorAlpha(col, alpha);
  g->SetLineColor(col);
  g->SetBit(TGraph::kClipFrame);
  if (!x.empty()) g->Draw("P");
  return g;
}

template <class TX, class TY>
TH2D* hist2(const std::vector<TX>& x, const std::vector<TY>& y, const Vec& xe, const Vec& ye, double w = 1.0) {
  auto* h = new TH2D(uid(), "", (int)xe.size() - 1, xe.data(), (int)ye.size() - 1, ye.data());
  for (size_t k = 0; k < x.size() && k < y.size(); ++k)
    if (std::isfinite((double)x[k]) && std::isfinite((double)y[k]))
      h->Fill(inLast(h->GetXaxis(), x[k]), inLast(h->GetYaxis(), y[k]), w);
  return h;
}
// 2D histogram with a log colour scale from its smallest non-zero bin; empty bins stay white.
void drawColz(TH2D* h, const TString& title, const TString& ztitle, bool logx = false, bool logy = false) {
  gStyle->SetPalette(kViridis);
  double mn = 1e300, mx = 0;
  for (int i = 1; i <= h->GetNbinsX(); ++i)
    for (int j = 1; j <= h->GetNbinsY(); ++j) {
      double c = h->GetBinContent(i, j);
      if (c > 0) {
        mn = std::min(mn, c);
        mx = std::max(mx, c);
      }
    }
  h->SetTitle(title);
  gPad->SetRightMargin(0.17);
  gPad->SetLogx(logx);
  gPad->SetLogy(logy);
  if (mx > 0) {
    h->SetMinimum(mn);
    h->SetMaximum(mx);
    h->GetZaxis()->SetTitle(ztitle);
    gPad->SetLogz();
    h->Draw("COLZ");
  } else {
    h->Draw("AXIS");
  }
}

// Binned efficiency like the old compute_efficiency(): bins with entries only,
// binomial errors.
void effPoints(const Vec& vals, const Vec& matched, const Vec& edges, Vec& x, Vec& y, Vec& ey) {
  int nb = (int)edges.size() - 1;
  Vec tot(nb, 0), pas(nb, 0);
  for (size_t k = 0; k < vals.size(); ++k) {
    double v = vals[k];
    if (!(v >= edges.front() && v <= edges.back())) continue;
    int b = int(std::upper_bound(edges.begin(), edges.end(), v) - edges.begin()) - 1;
    if (b >= nb) b = nb - 1;
    tot[b] += 1;
    pas[b] += matched[k];
  }
  x.clear(); y.clear(); ey.clear();
  for (int b = 0; b < nb; ++b) {
    if (tot[b] <= 0) continue;
    double e = pas[b] / tot[b];
    x.push_back(0.5 * (edges[b] + edges[b + 1]));
    y.push_back(e);
    ey.push_back(std::sqrt(e * (1 - e) / std::max(tot[b], 1.0)));
  }
}
TGraphErrors* effGraph(const Vec& x, const Vec& y, const Vec& ey, int col, int marker, int ls = 1) {
  Vec ex(x.size(), 0.);
  auto* g = new TGraphErrors((int)x.size(), x.data(), y.data(), ex.data(), ey.data());
  g->SetMarkerStyle(marker);
  g->SetMarkerColor(col);
  g->SetLineColor(col);
  g->SetLineStyle(ls);
  g->SetLineWidth(2);
  if (!x.empty()) g->Draw("PL");
  return g;
}

TString pyList(const std::set<int>& s) {   // [1, 2, 3]
  TString out = "[";
  for (auto it = s.begin(); it != s.end(); ++it) out += TString::Format("%s%d", it == s.begin() ? "" : ", ", *it);
  return out + "]";
}

// ---- reading ---------------------------------------------------------------------------
std::vector<TString> listNtuples(const TString& dir) {
  std::vector<std::pair<Long64_t, TString>> found;
  void* dp = gSystem->OpenDirectory(dir);
  if (!dp) return {};
  while (const char* e = gSystem->GetDirEntry(dp)) {
    TString name(e);
    Ssiz_t k = name.Index("_ntuple");
    if (k == kNPOS || !name.EndsWith(".root")) continue;
    TString num = name(k + 7, name.Length() - k - 7 - 5);
    if (num.IsNull() || !num.IsDigit()) continue;
    found.push_back({num.Atoll(), dir + "/" + name});
  }
  gSystem->FreeDirectory(dp);
  std::sort(found.begin(), found.end());   // numerically, like the old natural sort
  std::vector<TString> out;
  for (auto& f : found) {
    if (MAX_FILES_PER_INPUT > 0 && (int)out.size() >= MAX_FILES_PER_INPUT) break;
    out.push_back(f.second);
  }
  return out;
}

bool readInput(InputData& d) {
  std::vector<TString> files = listNtuples(d.dir);
  d.nFiles = (int)files.size();
  if (files.empty()) {
    Error("trackPlotter", "no <sample>_ntuple<N>.root files in %s", d.dir.Data());
    return false;
  }
  TChain ev("events"), sm("summary");
  for (auto& f : files) {
    ev.Add(f);
    sm.Add(f);
  }

  TTreeReader r(&ev);
  TTreeReaderValue<Bool_t> noHits(r, "no_hits"), detNoHits(r, "detected_no_hits"), nhUnavail(r, "nhits_unavailable"),
      useConed(r, "use_coned"), hasMuon(r, "has_muon");
  TTreeReaderValue<Int_t> nAll(r, "n_alltracks"), nDed(r, "n_deduped"), nSi(r, "n_sitracks"), nSeeds(r, "n_seeds"),
      muSiLinks(r, "muon_n_si_links"), seedMatch(r, "seed_match"), dispKind(r, "disp_kind");
  TTreeReaderValue<Double_t> seedTruthPt(r, "seed_truth_pt"), seedMatchPt(r, "seed_match_pt"),
      dispChi2(r, "disp_chi2_mean"), dispTruthPt(r, "disp_truth_pt"), dispRecoPt(r, "disp_reco_pt");
  TTreeReaderArray<Int_t> nDiag(r, "n_diag"), layCode(r, "lay_code"), layN(r, "lay_nhits");
  TTreeReaderArray<Double_t> muEta(r, "mu_eta"), muPt(r, "mu_pt"), muPhi(r, "mu_phi"), muTheta(r, "mu_theta_deg");
  TTreeReaderArray<Int_t> muMSi(r, "mu_matched_si"), muMAll(r, "mu_matched_all"), muNSi(r, "mu_n_si"), muNAll(r, "mu_n_all");
  TTreeReaderArray<Double_t> siChi2(r, "si_chi2red"), siEta(r, "si_eta"), siTruthPt(r, "si_truth_pt"),
      siRecoPt(r, "si_reco_pt"), siPtRes(r, "si_pt_res"), siPhi(r, "si_phi"), siD0(r, "si_d0"), siZ0(r, "si_z0"),
      siD0Res(r, "si_d0_res"), siZ0Res(r, "si_z0_res"), siDqop2(r, "si_dqop2"), siMuFrac(r, "si_mufrac");
  TTreeReaderArray<Int_t> siNbib(r, "si_nbib"), siPix(r, "si_npix"), siInner(r, "si_ninner"), siOuter(r, "si_nouter"),
      siTotal(r, "si_ntotal"), sihitCode(r, "sihit_code");
  TTreeReaderArray<Double_t> matChi2(r, "mat_chi2red");
  TTreeReaderArray<Int_t> matPix(r, "mat_npix"), matInner(r, "mat_ninner"), matOuter(r, "mat_nouter"), matTotal(r, "mat_ntotal");
  TTreeReaderArray<Int_t> rawNh(r, "raw_nhits");
  TTreeReaderArray<Float_t> rawChi2(r, "raw_chi2red"), rawPt(r, "raw_pt"), rawD0(r, "raw_d0"), rawZ0(r, "raw_z0");
  TTreeReaderArray<Float_t> sdPt(r, "sd_pt"), sdEta(r, "sd_eta"), zsZ(r, "zs_z"), zsR(r, "zs_r");
  TTreeReaderArray<Int_t> zsSys(r, "zs_sys"), dhKind(r, "dh_kind");
  TTreeReaderArray<Double_t> dhX(r, "dh_x"), dhY(r, "dh_y"), dhZ(r, "dh_z"), dhR(r, "dh_r");

  Long64_t zseamN = 0;
  while (r.Next()) {
    const Long64_t gi = r.GetCurrentEntry();   // index of the event in this input
    const bool nh = *noHits;
    if (!d.haveMode) {
      d.haveMode = true;
      d.noHits = nh;
      d.detectedNoHits = *detNoHits;
      d.nhitsUnavailable = *nhUnavail;
      d.useConed = *useConed;
    } else if (nh != d.noHits || (bool)*nhUnavail != d.nhitsUnavailable) {
      d.mixedModes = true;
    }
    ++d.nEvents;

    // [DIAG]: per-subdetector and merged hit-collection sizes, first 3 events.
    if (gi < 3 && !nh) {
      for (int k = 0; k < 7; ++k) {
        TString name = k < 6 ? TString(DIAG_NAMES[k]) + (*useConed ? "Coned" : "") : TString("MergedTrackerHits");
        TString n = nDiag[k] >= 0 ? TString::Format("%d", nDiag[k]) : TString("MISSING");
        d.diagLines.push_back(TString::Format("[DIAG] Event %lld %s: %s hits", gi, name.Data(), n.Data()));
      }
    }

    d.allMult.push_back(*nAll);
    d.dedupMult.push_back(*nDed);
    d.siMult.push_back(*nSi);
    d.seedMult.push_back(*nSeeds);
    if (*hasMuon) d.muonMatchedSi.push_back(*muSiLinks);
    if (!nh) {
      std::map<int, int> m;
      for (size_t k = 0; k < layCode.GetSize(); ++k) m[layCode[k]] = layN[k];
      d.layerNhits.push_back(m);
    }

    for (size_t k = 0; k < muEta.GetSize(); ++k) {
      d.effEta.push_back(muEta[k]);
      d.effPt.push_back(muPt[k]);
      d.effPhi.push_back(muPhi[k]);
      d.effTheta.push_back(muTheta[k]);
      d.effSi.push_back(muMSi[k]);
      d.effAll.push_back(muMAll[k]);
      if (muNSi[k] == 1) ++d.matchedNonrefit;
      if (muNAll[k] >= 1) ++d.matchedAllTracks;
    }

    for (size_t k = 0; k < siChi2.GetSize(); ++k) {
      d.chi2Nr.push_back(siChi2[k]);
      d.etaNr.push_back(siEta[k]);
      d.truthPtNr.push_back(siTruthPt[k]);
      d.recoPtNr.push_back(siRecoPt[k]);
      d.ptResNr.push_back(siPtRes[k]);
      d.phiNr.push_back(siPhi[k]);
      d.d0Nr.push_back(siD0[k]);
      d.z0Nr.push_back(siZ0[k]);
      if (std::isfinite(siD0Res[k])) {
        d.d0ResNr.push_back(siD0Res[k]);
        d.z0ResNr.push_back(siZ0Res[k]);
      }
      d.dqop2Nr.push_back(siDqop2[k]);
      // Without the hit collections only the total hit count exists.
      if (nh) {
        if (siTotal[k] > 0) d.totalNr.push_back(siTotal[k]);
      } else {
        d.muFracNr.push_back(siMuFrac[k]);
        d.nbibNr.push_back(siNbib[k]);
        d.pixNr.push_back(siPix[k]);
        d.innerNr.push_back(siInner[k]);
        d.outerNr.push_back(siOuter[k]);
        d.totalNr.push_back(siTotal[k]);
      }
    }
    for (size_t k = 0; k < sihitCode.GetSize(); ++k) ++d.layerHitsNr[sihitCode[k] / 64][sihitCode[k] % 64];

    for (size_t k = 0; k < matChi2.GetSize(); ++k) {
      if (std::isfinite(matChi2[k])) d.chi2All.push_back(matChi2[k]);
      if (matTotal[k] >= 0) {
        if (!nh) {
          d.pixAll.push_back(matPix[k]);
          d.innerAll.push_back(matInner[k]);
          d.outerAll.push_back(matOuter[k]);
        }
        d.totalAll.push_back(matTotal[k]);
      }
    }

    for (size_t k = 0; k < rawNh.GetSize(); ++k) {
      if (rawNh[k] > 0) d.rawNhits.push_back(rawNh[k]);
      if (std::isfinite(rawChi2[k])) d.rawChi2.push_back(rawChi2[k]);
      if (std::isfinite(rawPt[k])) d.rawPt.push_back(rawPt[k]);
      if (std::isfinite(rawD0[k])) {
        d.rawD0.push_back(rawD0[k]);
        d.rawZ0.push_back(rawZ0[k]);
      }
    }
    for (size_t k = 0; k < sdPt.GetSize(); ++k) {
      d.seedPt.push_back(sdPt[k]);
      d.seedEta.push_back(sdEta[k]);
    }
    if (*seedMatch > 0) {
      d.seedBest.push_back({*seedTruthPt, *seedMatchPt});
      if (*seedMatch == 1) ++d.nSeedHit;
      else ++d.nSeedDr;
    }

    // Same cap rule as the old single pass: an event's hits are added while
    // the running total is below ZSEAM_MAX_HITS.
    if (zseamN < ZSEAM_MAX_HITS) {
      for (size_t k = 0; k < zsZ.GetSize(); ++k) {
        d.zsZ.push_back(zsZ[k]);
        d.zsR.push_back(zsR[k]);
        d.zsSys.push_back(zsSys[k]);
      }
      zseamN += zsZ.GetSize();
    }

    const int kind = *dispKind;
    if ((kind & 1) || ((kind & 2) && d.allRz.size() < 10)) {
      Display ds;
      ds.event = gi;
      ds.chi2Mean = *dispChi2;
      ds.truthPt = *dispTruthPt;
      ds.recoPt = *dispRecoPt;
      for (size_t k = 0; k < dhKind.GetSize(); ++k) {
        int c = dhKind[k];
        if (c < 0 || c > 2) continue;
        ds.x[c].push_back(dhX[k]);
        ds.y[c].push_back(dhY[k]);
        ds.z[c].push_back(dhZ[k]);
        ds.r[c].push_back(dhR[k]);
      }
      if (kind & 1) d.displays.push_back(ds);
      if ((kind & 2) && d.allRz.size() < 10) d.allRz.push_back(ds);
    }
  }
  if (r.GetEntryStatus() != TTreeReader::kEntryNotFound && r.GetEntryStatus() != TTreeReader::kEntryBeyondEnd &&
      r.GetEntryStatus() != TTreeReader::kEntryValid)
    Error("trackPlotter", "reading %s stopped early (TTreeReader status %d)", d.dir.Data(), (int)r.GetEntryStatus());

  TTreeReader rs(&sm);
  TTreeReaderValue<Int_t> sPdg(rs, "signal_pdg"), sMinHits(rs, "seed_min_hits");
  TTreeReaderValue<Double_t> sMaxDr(rs, "seed_max_dr"), sZsHalf(rs, "zseam_collect_half");
  TTreeReaderArray<Int_t> occ(rs, "occ"), idCode(rs, "id_code"), sampSys(rs, "samp_sys"), sampSide(rs, "samp_side"),
      sampLayer(rs, "samp_layer");
  TTreeReaderArray<Double_t> idCount(rs, "id_count");   // whole numbers, stored as double
  TTreeReaderArray<Float_t> sampR(rs, "samp_r"), sampZ(rs, "samp_z");
  TTreeReaderArray<Double_t> sampU(rs, "samp_u");
  d.occ.assign(OCC_NZ * OCC_NR, 0.);
  bool first = true;
  while (rs.Next()) {
    if (first) {
      d.signalPdg = *sPdg;
      d.seedMinHits = *sMinHits;
      d.seedMaxDr = *sMaxDr;
      d.zseamCollectHalf = *sZsHalf;
      first = false;
    }
    for (size_t k = 0; k < occ.GetSize() && k < d.occ.size(); ++k) d.occ[k] += occ[k];
    for (size_t k = 0; k < idCode.GetSize(); ++k) {
      int c = idCode[k];
      const Long64_t n = std::llround(idCount[k]);
      d.idCounts[{c & 0x1F, (c >> 5) & 0x3, (c >> 7) & 0x3F}] += n;
      d.idSeen[c & 0x1F] += n;
    }
    for (size_t k = 0; k < sampSys.GetSize(); ++k)
      d.idSample[sampSys[k]].push_back({sampU[k], sampR[k], sampZ[k], sampSide[k], sampLayer[k]});
  }
  // The per-file samples merge into one uniform sample: keep the smallest u.
  for (auto& kv : d.idSample) {
    auto& v = kv.second;
    if ((int)v.size() > ID_DIAG_TARGET_HITS_PER_SYSTEM) {
      std::nth_element(v.begin(), v.begin() + ID_DIAG_TARGET_HITS_PER_SYSTEM, v.end(),
                       [](const IdSample& a, const IdSample& b) { return a.u < b.u; });
      v.resize(ID_DIAG_TARGET_HITS_PER_SYSTEM);
    }
  }
  return true;
}

void printHeader(const InputData& d, const TString& plotdir) {
  printf("Processing: %s (%d file%s)\n", d.dir.Data(), d.nFiles, d.nFiles == 1 ? "" : "s");
  printf("  USE_CONED = %s (from the ntuples)\n", d.useConed ? "True" : "False");
  if (d.mixedModes)
    printf("  WARNING: the files of this entry differ in which hit collections they have; each was read in "
           "its own mode and the plots mix them.\n");
  if (d.noHits) {
    printf("  NO_HITS = True (from %s): MergedTrackerHits / MergedTrackerHitsRelations absent - hit-level plots "
           "disabled (subsystem split, muon/BIB split, per-layer, event displays, occupancy).\n",
           d.detectedNoHits ? "collection autodetect" : "env var");
    printf("  per-track hit counts from: %s%s\n", d.nhitsUnavailable ? "unavailable" : "trackerHits",
           d.nhitsUnavailable ? "  (none available: hit-count plots will be empty and the multiplicity "
                                "counters count every track)"
                              : "");
  } else if (d.detectedNoHits) {
    printf("  WARNING: NO_HITS forced off by the env var, but this file has no MergedTrackerHits / "
           "MergedTrackerHitsRelations - the hit-level code will fail on it.\n");
  }
  for (auto& l : d.diagLines) printf("%s\n", l.Data());
  printf("  plots -> %s/\n", plotdir.Data());
}

// ---- event hit maps ---------------------------------------------------------------------
void drawDetectorRZ() {
  auto L = [](double x1, double y1, double x2, double y2, int col, double alpha) {
    auto* l = new TLine(x1, y1, x2, y2);
    l->SetLineColorAlpha(col, alpha);
    l->Draw();
  };
  int cv = TColor::GetColor("#348ec2"), ci = TColor::GetColor("#34b18f"), co = TColor::GetColor("#dd7e33");
  for (double r : vb_r) {
    L(-vb_l / 2, r, vb_l / 2, r, cv, 1);
    L(-vb_l / 2, r + 2, vb_l / 2, r + 2, cv, 1);
  }
  for (int j = 0; j < 4; ++j) {
    double z = ve_z[j];
    L(z, ve_h[j], z, 125, cv, .5);
    L(-z, ve_h[j], -z, 125, cv, .5);
    L(z + 2, ve_h[j], z + 2, 125, cv, .5);
    L(-z + 2, ve_h[j], -z + 2, 125, cv, .5);
  }
  for (int j = 0; j < 3; ++j) L(-ib_l[j] / 2, ib_r[j], ib_l[j] / 2, ib_r[j], ci, 1);
  for (int j = 0; j < 7; ++j) {
    double top = j == 0 ? 410 : 570;
    L(ie_z[j], ie_h(j), ie_z[j], top, ci, .5);
    L(-ie_z[j], ie_h(j), -ie_z[j], top, ci, .5);
  }
  for (double r : ob_r) L(-ob_l / 2, r, ob_l / 2, r, co, 1);
  for (double z : oe_z) {
    L(z, 600, z, 1430, co, .5);
    L(-z, 600, -z, 1430, co, .5);
  }
}

// Title lines above the frame, like the old multi-line titles.
void hitMapTitle(const char* view, const Display& d) {
  std::vector<TString> lines = {TString::Format("Non-Refitted %s Hit Map", view)};
  if (std::isfinite(d.chi2Mean)) lines.push_back(TString::Format("#chi^{2}/Ndf = %.2f", d.chi2Mean));
  if (std::isfinite(d.truthPt)) lines.push_back(TString::Format("Truth Muon p_{T} = %.1f GeV", d.truthPt));
  if (std::isfinite(d.recoPt)) lines.push_back(TString::Format("Reconstructed Track p_{T} = %.1f GeV", d.recoPt));
  TLatex t;
  t.SetTextAlign(22);
  t.SetTextSize(0.035);
  for (size_t k = 0; k < lines.size(); ++k) t.DrawLatexNDC(0.55, 0.965 - 0.04 * k, lines[k]);
}

void scatterHits(const Display& d, bool rz, TLegend* leg) {
  const char* names[3] = {"Muon-matched hits in track", "Non-muon hits", "Muon hits (not in track)"};
  int cols[3] = {steelblue(), orange(), kRed};
  double alpha[3] = {0.6, 0.6, 0.7};
  int markers[3] = {20, 20, 22};
  for (int k = 0; k < 3; ++k) {
    if (d.x[k].empty()) continue;
    TGraph* g = rz ? scatter(d.z[k], d.r[k], cols[k], alpha[k], markers[k], 0.8)
                   : scatter(d.x[k], d.y[k], cols[k], alpha[k], markers[k], 0.8);
    leg->AddEntry(g, names[k], "p");
  }
}

// r-z map. allRzStyle: the "first 10 events" variant (fit through the origin,
// fixed axis ranges).
TCanvas* eventRZ(const Display& d, bool allRzStyle) {
  auto* c = canvas(600, 600);
  c->SetTopMargin(0.2);
  c->SetBit(TGraph::kClipFrame);
  const Vec& z = d.z[0];
  const Vec& r = d.r[0];
  bool fit = !z.empty();
  double x1 = -RZ_ZLIM, x2 = RZ_ZLIM;
  if (!fit && !allRzStyle) {   // no limits were set: the whole tracker
    x1 = -2300;
    x2 = 2300;
  }
  c->DrawFrame(x1, 0, x2, 1600, ";z [mm];r [mm]");
  drawDetectorRZ();
  auto* leg = legend(0.15, 0.62, 0.6, 0.79, 0.028);
  scatterHits(d, true, leg);
  if (fit) {
    double a = 0, b = 0, zend = 0;
    size_t n = z.size();
    if (allRzStyle) {   // through the origin, up to the farthest hit in z
      double zr = 0, zz = 0;
      for (size_t k = 0; k < n; ++k) {
        zr += z[k] * r[k];
        zz += z[k] * z[k];
        if (std::fabs(z[k]) > std::fabs(zend)) zend = z[k];
      }
      a = zz > 0 ? zr / zz : 0;
    } else {   // least squares, as np.polyfit(z, r, 1)
      double sz = 0, sr = 0, szz = 0, szr = 0;
      for (size_t k = 0; k < n; ++k) {
        sz += z[k];
        sr += r[k];
        szz += z[k] * z[k];
        szr += z[k] * r[k];
      }
      double den = n * szz - sz * sz;
      if (n >= 2 && std::fabs(den) > 1e-12) {
        a = (n * szr - sz * sr) / den;
        b = (sr - a * sz) / n;
      } else {   // one z only: np.polyfit's minimum-norm solution
        double z0 = sz / n, r0 = sr / n;
        a = r0 * z0 / (z0 * z0 + 1);
        b = r0 / (z0 * z0 + 1);
      }
    }
    Vec lz = allRzStyle ? linspace(0, zend, 200) : linspace(x1, x2, 200);
    Vec lr(lz.size());
    for (size_t k = 0; k < lz.size(); ++k) lr[k] = a * lz[k] + b;
    auto* g = new TGraph((int)lz.size(), lz.data(), lr.data());
    g->SetLineColor(mpl(0));
    g->SetBit(TGraph::kClipFrame);
    g->Draw("L");
    leg->AddEntry(g, TString::Format("fit (N=%zu)", n), "l");
  }
  leg->Draw();
  hitMapTitle("r-z", d);
  return c;
}

TCanvas* eventXY(const Display& d) {
  // Square frame (480 x 480 px) so the layer circles stay round.
  auto* c = canvas(600, 706);
  c->SetLeftMargin(0.14);
  c->SetRightMargin(0.06);
  c->SetTopMargin(0.2);
  c->SetBottomMargin(0.12);
  c->SetBit(TGraph::kClipFrame);
  c->DrawFrame(-1400, -1400, 1400, 1400, ";x [mm];y [mm]");
  auto circles = [](const double* radii, int n, int col) {
    for (int k = 0; k < n; ++k) {
      auto* e = new TEllipse(0, 0, radii[k], radii[k]);
      e->SetFillStyle(0);
      e->SetLineColor(col);
      e->SetLineWidth(1);
      e->Draw();
    }
  };
  circles(vb_r, 4, TColor::GetColor("#348ec2"));
  circles(ib_r, 3, TColor::GetColor("#34b18f"));
  circles(ob_r, 3, TColor::GetColor("#dd7e33"));
  auto* leg = legend(0.15, 0.62, 0.6, 0.79, 0.028);
  scatterHits(d, false, leg);
  leg->Draw();
  hitMapTitle("x-y", d);
  return c;
}

void drawDisplays(const InputData& d, const TString& plotdir) {
  // Single-event files first: a multi-page PDF has to stay open while its pages are added.
  for (auto& ds : d.displays) {
    save(eventRZ(ds, false), TString::Format("%s/EventRZ_%lld.pdf", plotdir.Data(), ds.event));
    if (!ds.x[1].empty()) save(eventXY(ds), TString::Format("%s/EventXY_%lld.pdf", plotdir.Data(), ds.event));
  }
  for (auto& ds : d.allRz) save(eventRZ(ds, true), TString::Format("%s/All_EventRZ_%lld.pdf", plotdir.Data(), ds.event));

  auto multipage = [](const std::vector<Display>& list, bool allRzStyle, const TString& path) {
    for (size_t k = 0; k < list.size(); ++k) {
      TCanvas* c = eventRZ(list[k], allRzStyle);
      TString suffix = list.size() == 1 ? "" : (k == 0 ? "(" : (k + 1 == list.size() ? ")" : ""));
      c->Print(path + suffix, "pdf");
      delete c;
    }
  };
  multipage(d.displays, false, plotdir + "/rz_hit_maps_all_events.pdf");
  multipage(d.allRz, true, plotdir + "/all_rz_hit_maps_muon_matched.pdf");
}

// ---- per-input plots ---------------------------------------------------------------------
void plotHitsPanels(const TString& path, const std::vector<const Vec*>& data, const std::vector<int>& cols,
                    const std::vector<TString>& titles, const std::vector<int>& nbins) {
  bool any = false;
  for (auto* v : data) any |= !v->empty();
  if (!any) return;
  auto* c = canvas(1000, 800);
  grid(c, 2, 2);
  for (int k = 0; k < 4; ++k) {
    c->cd(k + 1);
    if (data[k]->empty()) {
      emptyPanel(titles[k] + " (no hit-level info in this file)");
      continue;
    }
    auto* h = new TH1D(uid(), titles[k] + ";Number of hits;Tracks", nbins[k], 0, nbins[k]);
    fillH(h, *data[k]);
    barStyle(h, cols[k]);
    h->SetMinimum(0);
    h->Draw("HIST");
    auto* l = legend(0.55, 0.72, 0.94, 0.88);
    addLines(l, h, "f", {TString::Format("Mean: %.2f", vmean(*data[k])), TString::Format("Median: %.1f", vmedian(*data[k]))});
    l->Draw();
  }
  save(c, path);
}

// Muon-matched-track distribution, filled salmon, normalised to the number of tracks.
void plotTrackDist(const Vec& v, const Vec& edges, const TString& title, bool logx, bool logy, const TString& path) {
  if (v.empty()) return;
  auto* c = canvas();
  auto* h = hist(v, edges, kPerEntry, title);
  barStyle(h, salmon(), 0.6);
  gPad->SetLogx(logx);
  drawAll({h}, logy);
  auto* l = legend(0.7, 0.8, 0.94, 0.88);
  l->AddEntry(h, "Non-Refitted", "f");
  l->Draw();
  save(c, path);
}

void plotEntry(const InputData& d, const TString& plotdir) {
  const Long64_t i = d.nEvents;
  const TString dir = plotdir + "/";
  if (!d.noHits) drawDisplays(d, plotdir);

  // --- muon-matched track multiplicity
  if (!d.muonMatchedSi.empty()) {
    int maxm = (int)vmax(d.muonMatchedSi);
    auto* c = canvas();
    auto* h = hist(d.muonMatchedSi, arange(-0.5, maxm + 1.5, 1), kUnit,
                   "Muon-Matched Track Multiplicity per Event;Number of muon-matched tracks per event;Fraction of events");
    stepStyle(h, mpl(0));
    drawAll({h}, true);
    auto* l = legend(0.6, 0.82, 0.94, 0.88);
    l->AddEntry(h, "SiTracks (non-refitted)", "l");
    l->Draw();
    save(c, dir + "muon_matched_track_mult_per_event.pdf");
  }

  // --- cutflow: tracks surviving each reconstruction stage
  if (!d.siMult.empty()) {
    const char* names[3] = {"AllTracks", "DedupedTracks", "SiTracks"};
    const Vec* lists[3] = {&d.allMult, &d.dedupMult, &d.siMult};
    double totals[3], means[3], stds[3];
    for (int k = 0; k < 3; ++k) {
      totals[k] = 0;
      for (double x : *lists[k]) totals[k] += x;
      means[k] = vmean(*lists[k]);
      stds[k] = vstd(*lists[k]);
    }
    printf("Track cutflow (totals across all events; %% relative to prev / AllTracks):\n");
    double prev = totals[0], firstT = totals[0] > 0 ? totals[0] : 1;
    for (int k = 0; k < 3; ++k) {
      printf("  %-20s total=%8.0f  mean/ev=%7.2f   vs prev: %5.1f%%   vs AllTracks: %5.1f%%\n", names[k], totals[k],
             means[k], prev > 0 ? 100 * totals[k] / prev : 0.0, firstT > 0 ? 100 * totals[k] / firstT : 0.0);
      prev = totals[k];
    }
    printf("Tracks per event (track collections: tracks with hits; SeedTracks: seeds with a track state):\n");
    const char* pnames[4] = {"AllTracks", "DedupedTracks", "SiTracks", "SeedTracks"};
    const Vec* plists[4] = {&d.allMult, &d.dedupMult, &d.siMult, &d.seedMult};
    for (int k = 0; k < 4; ++k)
      printf("  %-20s mean=%10.2f  median=%10.1f  std dev=%10.2f\n", pnames[k], vmean(*plists[k]),
             vmedian(*plists[k]), vstd(*plists[k]));

    auto* c = canvas(1300, 500);
    grid(c, 2, 1);
    c->cd(1);
    auto* ht = new TH1D(uid(), "Track Cutflow: total surviving per stage;;Total tracks (all events)", 3, 0, 3);
    auto* hm = new TH1D(uid(), "Track Cutflow: per-event average;;Mean tracks / event (#pmRMS)", 3, 0, 3);
    double tmax = 0, mmax = 0;
    for (int k = 0; k < 3; ++k) {
      ht->SetBinContent(k + 1, totals[k]);
      hm->SetBinContent(k + 1, means[k]);
      hm->SetBinError(k + 1, stds[k]);
      ht->GetXaxis()->SetBinLabel(k + 1, names[k]);
      hm->GetXaxis()->SetBinLabel(k + 1, names[k]);
      tmax = std::max(tmax, totals[k]);
      mmax = std::max(mmax, means[k] + stds[k]);
    }
    TLatex t;
    t.SetTextAlign(21);
    t.SetTextSize(0.035);
    barStyle(ht, steelblue());
    ht->SetBarWidth(0.8);
    ht->SetBarOffset(0.1);
    ht->SetMinimum(0);
    ht->SetMaximum(tmax > 0 ? tmax * 1.15 : 1);
    ht->Draw("BAR");
    for (int k = 0; k < 3; ++k) t.DrawLatex(k + 0.5, totals[k], TString::Format("%.0f", totals[k]));
    gPad->SetGridy();
    c->cd(2);
    barStyle(hm, salmon());
    hm->SetBarWidth(0.8);
    hm->SetBarOffset(0.1);
    hm->SetMinimum(0);
    hm->SetMaximum(mmax > 0 ? mmax * 1.15 : 1);
    hm->SetMarkerSize(0);
    hm->Draw("BAR");
    hm->Draw("E1 SAME");
    for (int k = 0; k < 3; ++k) t.DrawLatex(k + 0.5, means[k], TString::Format("%.2f", means[k]));
    gPad->SetGridy();
    save(c, dir + "track_cutflow.pdf");

    double lo = 1e300, hi = 0;
    for (int k = 0; k < 3; ++k)
      for (double x : *lists[k]) {
        if (x > 0) lo = std::min(lo, x);
        hi = std::max(hi, x);
      }
    lo = lo < 1e300 ? std::max(1.0, lo) : 1;
    Vec edges = logspace(lo, hi + 1, 51);
    auto* c2 = canvas();
    gPad->SetLogx();
    std::vector<TH1*> hs;
    auto* l = legend(0.55, 0.74, 0.94, 0.88);
    for (int k = 0; k < 3; ++k) {
      if (lists[k]->empty()) continue;
      auto* h = hist(*lists[k], edges, kUnit, "Track Multiplicity per Event;Track multiplicity per event;Fraction of events");
      stepStyle(h, mpl(k));
      hs.push_back(h);
      l->AddEntry(h, TString::Format("%s  (mean=%.3g)", names[k], vmean(*lists[k])), "l");
    }
    drawAll(hs, false);
    l->Draw();
    save(c2, dir + "track_multiplicity_per_event.pdf");
  }

  // --- hits per event in each tracker layer
  if (!d.layerNhits.empty()) {
    printf("Hits per event in each layer (MergedTrackerHits; an endcap layer counts both sides):\n");
    std::set<int> codes;
    for (auto& m : d.layerNhits)
      for (auto& kv : m) codes.insert(kv.first);
    for (int code : codes) {
      Vec counts;
      for (auto& m : d.layerNhits) {
        auto it = m.find(code);
        counts.push_back(it == m.end() ? 0 : it->second);
      }
      printf("  %-10s layer %2d  mean=%10.2f  median=%10.1f  std dev=%10.2f\n", relname(code / 64), code % 64,
             vmean(counts), vmedian(counts), vstd(counts));
    }
  }

  double effNr = i > 0 ? double(d.matchedNonrefit) / i : 0.0, effAll = i > 0 ? double(d.matchedAllTracks) / i : 0.0;
  printf("AllTracks (>=1 truth-matched) efficiency: %ld/%lld (%.1f%%)\n", d.matchedAllTracks, i, 100 * effAll);
  printf("SiTracks (non-refitted) efficiency:       %ld/%lld (%.1f%%)\n", d.matchedNonrefit, i, 100 * effNr);

  // --- AllTracks raw distributions with the AllTracks -> SiTracks filter cuts
  if (!d.rawNhits.empty() || !d.rawPt.empty() || !d.rawD0.empty()) {
    auto* c = canvas(1200, 900);
    grid(c, 2, 2, "AllTracks raw distributions with AllTracks#rightarrowSiTracks filter cuts");
    c->cd(1);
    if (!d.rawNhits.empty()) {
      double nhmax = std::max(vmax(d.rawNhits), 20.0);
      auto* h = hist(d.rawNhits, arange(-0.5, nhmax + 1.5, 1), kCounts, "AllTracks: NHits per track (raw);NHits per AllTrack;Tracks");
      barStyle(h, steelblue());
      drawAll({h}, true);
      vline(PRF_NHITS_MIN - 0.5, kRed);
      auto* l = legend(0.5, 0.66, 0.94, 0.88);
      addLines(l, h, "f", {TString::Format("N=%zu", d.rawNhits.size()), TString::Format("median=%.1f", vmedian(d.rawNhits)),
                           TString::Format("mean=%.2f", vmean(d.rawNhits))});
      auto* cut = new TLine();
      cut->SetLineColor(kRed);
      cut->SetLineStyle(2);
      l->AddEntry(cut, TString::Format("pre-refit cut: NHits>=%d", PRF_NHITS_MIN), "l");
      l->Draw();
    } else {
      emptyPanel("AllTracks: NHits per track (no hit counts in this file)");
    }
    c->cd(2);
    if (!d.rawChi2.empty()) {
      gPad->SetLogx();
      auto* h = hist(d.rawChi2, logspace(1e-4, 1e6, 60), kCounts, "AllTracks: chi2/Ndf (raw);#chi^{2} / Ndf;Tracks");
      barStyle(h, steelblue());
      drawAll({h}, true);
      auto* l = legend(0.6, 0.76, 0.94, 0.88);
      addLines(l, h, "f", {TString::Format("N=%zu", d.rawChi2.size()), TString::Format("median=%.2e", vmedian(d.rawChi2))});
      l->Draw();
    }
    c->cd(3);
    if (!d.rawPt.empty()) {
      gPad->SetLogx();
      double ptmin = std::max(vmin(d.rawPt) * 0.5, 1e-4), ptmax = vmax(d.rawPt) * 1.5;
      auto* h = hist(d.rawPt, logspace(ptmin, ptmax, 60), kCounts, "AllTracks: reco p_{T} (raw);Reco p_{T} [GeV];Tracks");
      barStyle(h, steelblue());
      drawAll({h}, true);
      vline(FILT_MIN_PT, kRed);
      auto* l = legend(0.55, 0.72, 0.94, 0.88);
      addLines(l, h, "f", {TString::Format("N=%zu", d.rawPt.size()), TString::Format("median=%.2g GeV", vmedian(d.rawPt))});
      auto* cut = new TLine();
      cut->SetLineColor(kRed);
      cut->SetLineStyle(2);
      l->AddEntry(cut, TString::Format("filter cut: pT>=%g GeV", FILT_MIN_PT), "l");
      l->Draw();
    }
    c->cd(4);
    if (!d.rawD0.empty() && !d.rawZ0.empty()) {
      double ext = 2 * FILT_MAX_D0;
      for (float x : d.rawD0) ext = std::max(ext, (double)std::fabs(x));
      for (float x : d.rawZ0) ext = std::max(ext, (double)std::fabs(x));
      Vec edges = linspace(-ext, ext, 81);
      auto* hd = hist(d.rawD0, edges, kCounts, "AllTracks: d_{0} and z_{0} (raw);d_{0} / z_{0} [mm];Tracks");
      auto* hz = hist(d.rawZ0, edges);
      stepStyle(hd, steelblue());
      stepStyle(hz, salmon());
      drawAll({hd, hz}, true);
      vline(-FILT_MAX_D0, kRed, 2, 1, 0.7);
      vline(FILT_MAX_D0, kRed, 2, 1, 0.7);
      long nd = 0, nz = 0;
      for (float x : d.rawD0) nd += std::fabs(x) > FILT_MAX_D0;
      for (float x : d.rawZ0) nz += std::fabs(x) > FILT_MAX_Z0;
      auto* l = legend(0.17, 0.72, 0.94, 0.88, 0.028);
      l->AddEntry(hd, TString::Format("D0  median=%.1f, |D0|>100: %ld/%zu", vmedian(d.rawD0), nd, d.rawD0.size()), "l");
      l->AddEntry(hz, TString::Format("Z0  median=%.1f, |Z0|>100: %ld/%zu", vmedian(d.rawZ0), nz, d.rawZ0.size()), "l");
      auto* cut = new TLine();
      cut->SetLineColor(kRed);
      cut->SetLineStyle(2);
      l->AddEntry(cut, TString::Format("filter cut: |D0|,|Z0|<%g mm", FILT_MAX_D0), "l");
      l->Draw();
    }
    save(c, dir + "alltracks_raw_filter_diagnostic.pdf");

    // Cut-survivor summary (non-exclusive)
    size_t N = std::max({d.rawNhits.size(), d.rawD0.size(), d.rawPt.size()});
    long failPt = 0, failD0 = 0, failZ0 = 0, failNh = 0;
    for (float x : d.rawPt) failPt += x < FILT_MIN_PT;
    for (float x : d.rawD0) failD0 += std::fabs(x) > FILT_MAX_D0;
    for (float x : d.rawZ0) failZ0 += std::fabs(x) > FILT_MAX_Z0;
    for (float x : d.rawNhits) failNh += x < PRF_NHITS_MIN;
    double Nd = std::max<double>(N, 1);
    printf("[AllTracks raw] N=%zu\n", N);
    printf("  fail pT < %g GeV: %ld/%zu (%.1f%%)\n", FILT_MIN_PT, failPt, N, 100 * failPt / Nd);
    printf("  fail |D0| > %g mm:  %ld/%zu (%.1f%%)\n", FILT_MAX_D0, failD0, N, 100 * failD0 / Nd);
    printf("  fail |Z0| > %g mm:  %ld/%zu (%.1f%%)\n", FILT_MAX_Z0, failZ0, N, 100 * failZ0 / Nd);
    if (!d.rawNhits.empty())
      printf("  fail NHits < %d (pre-refit): %ld/%zu (%.1f%%)\n", PRF_NHITS_MIN, failNh, d.rawNhits.size(),
             100. * failNh / std::max<double>(d.rawNhits.size(), 1));
    else
      printf("  fail NHits < %d (pre-refit): n/a (no hit counts in this file)\n", PRF_NHITS_MIN);
  }

  // --- pixel / inner / outer / total hits on the muon-matched tracks
  plotHitsPanels(dir + "all_hits_combined_sitracks.pdf", {&d.pixNr, &d.innerNr, &d.outerNr, &d.totalNr},
                 {skyblue(), lightgreen(), salmon(), green()},
                 {"Pixel Hits (SiTracks)", "Inner Hits (SiTracks)", "Outer Hits (SiTracks)", "Total Hits (SiTracks)"},
                 {10, 5, 5, 15});
  plotHitsPanels(dir + "all_hits_combined_alltracks.pdf", {&d.pixAll, &d.innerAll, &d.outerAll, &d.totalAll},
                 {skyblue(), lightgreen(), salmon(), green()},
                 {"Pixel Hits (AllTracks)", "Inner Hits (AllTracks)", "Outer Hits (AllTracks)", "Total Hits (AllTracks)"},
                 {10, 5, 5, 15});

  // --- chi2/Ndf of the muon-matched tracks
  if (!d.chi2Nr.empty() || !d.chi2All.empty()) {
    auto* c = canvas();
    gPad->SetLogx();
    Vec edges = logspace(1e-2, 1e6, 60);
    std::vector<TH1*> hs;
    auto* l = legend(0.7, 0.78, 0.94, 0.88);
    if (!d.chi2All.empty()) {
      auto* h = hist(d.chi2All, edges, kPerEntry, "Matched Muon Tracks Chi2 Reduced;Chi2 Reduced;Fraction of tracks");
      stepStyle(h, purple());
      hs.push_back(h);
      l->AddEntry(h, "AllTracks", "l");
    }
    if (!d.chi2Nr.empty()) {
      auto* h = hist(d.chi2Nr, edges, kPerEntry, "Matched Muon Tracks Chi2 Reduced;Chi2 Reduced;Fraction of tracks");
      barStyle(h, steelblue(), 0.6);
      hs.push_back(h);
      l->AddEntry(h, "Non-Refitted", "f");
    }
    drawAll(hs, true);
    l->Draw();
    save(c, dir + "chi2_reduced.pdf");
  }

  // --- non-refitted 2D summaries
  if (!d.chi2Nr.empty()) {
    double ptmax = 2 * vmax(d.truthPtNr);
    {
      auto* c = canvas();
      auto* h = hist2(d.truthPtNr, d.recoPtNr, linspace(0, ptmax, 51), linspace(0, ptmax, 51));
      drawColz(h, "Truth vs Reconstructed p_{T} (Non-Refitted);Truth Muon p_{T} [GeV];Reconstructed Track p_{T} [GeV]", "Counts");
      auto* diag = new TLine(0, 0, ptmax, ptmax);
      diag->SetLineColor(kRed);
      diag->SetLineStyle(2);
      diag->Draw();
      auto* l = legend(0.17, 0.82, 0.5, 0.88);
      l->AddEntry(diag, "truth = reco", "l");
      l->Draw();
      save(c, dir + "truth_vs_reco_pt_nonrefit.pdf");
    }
    {
      auto* c = canvas();
      auto* h = hist2(d.chi2Nr, d.etaNr, logspace(1e-2, 1e4, 60), linspace(-3, 3, 61));
      drawColz(h, "Chi2 vs #eta (Non-Refitted);Chi2 Reduced;Track #eta", "Counts", true, false);
      save(c, dir + "chi2_vs_eta_nonrefit_hist2d.pdf");
    }
    {
      auto* c = canvas();
      auto* h = hist2(d.truthPtNr, d.chi2Nr, linspace(0, 2500, 26), logspace(1e-2, 1e4, 60));
      drawColz(h, "p_{T} vs Chi2 Reduced (Non-Refitted);Truth Muon p_{T} [GeV];Chi2 Reduced", "Counts", false, true);
      save(c, dir + "pt_vs_chi2_nonrefit_hist2d.pdf");
    }
  }
  // These need the per-hit muon/BIB classification.
  if (!d.chi2Nr.empty() && !d.muFracNr.empty()) {
    {
      auto* c = canvas();
      auto* h = hist2(d.truthPtNr, d.muFracNr, linspace(0, 2500, 26), linspace(0, 1, 21));
      drawColz(h, "p_{T} vs Fraction of Muon Hits (Non-Refitted);Truth Muon p_{T} [GeV];Fraction of Muon Hits", "Counts");
      save(c, dir + "pt_vs_muonhitfraction_nonrefit_hist2d.pdf");
    }
    {
      auto* c = canvas();
      auto* h = hist2(d.muFracNr, d.chi2Nr, linspace(0, 1, 21), logspace(1e-2, 1e4, 60));
      drawColz(h, "Fraction of Muon Hits vs Chi2 Reduced (Non-Refitted);Fraction of Muon Hits;Chi2 Reduced", "Counts", false, true);
      save(c, dir + "muonhitfraction_vs_chi2_nonrefit_hist2d.pdf");
    }
    {
      auto* c = canvas();
      auto* h = hist2(d.nbibNr, d.chi2Nr, linspace(0, 10, 11), logspace(1e-2, 1e4, 60));
      drawColz(h, "# BIB Hits vs Chi2 Reduced (Non-Refitted);# BIB Hits in Muon Matched Track;Chi2 Reduced", "Counts", false, true);
      save(c, dir + "numberBIBhits_vs_chi2_nonrefit_hist2d.pdf");
    }
    {
      auto* c = canvas();
      auto* h = hist2(d.muFracNr, d.ptResNr, linspace(0, 1, 21), linspace(-1, 1, 81), 1.0 / d.ptResNr.size());
      drawColz(h, "p_{T} Resolution vs Fraction of Muon Hits (Non-Refitted);Fraction of Muon Hits;"
                  "p_{T} Resolution (p_{T}^{reco} - p_{T}^{truth}) / p_{T}^{truth}",
               "Fraction of tracks");
      save(c, dir + "pt_resolution_vs_muonhitfraction_nonrefit_hist2d.pdf");
    }
    {
      auto* c = canvas();
      double mn = std::max(vmin(d.dqop2Nr), 1e-12);
      auto* h = hist2(d.muFracNr, d.dqop2Nr, linspace(0, 1, 21), logspace(mn, vmax(d.dqop2Nr) * 1.1, 81),
                      1.0 / d.dqop2Nr.size());
      drawColz(h, "#sigma(q/p)^{2} vs Fraction of Muon Hits (Non-Refitted);Fraction of Muon Hits;"
                  "#sigma(q/p)^{2} = (q/p_{reco} - q/p_{truth})^{2} [GeV^{-2}]",
               "Fraction of tracks", false, true);
      save(c, dir + "delta_qoverp_sq_vs_muonhitfraction_nonrefit_hist2d.pdf");
    }
  }

  // --- seeds
  if (!d.seedPt.empty()) {
    {
      auto* c = canvas();
      gPad->SetLogx();
      auto* h = hist(d.seedPt, logspace(1, 5000, 81), kCounts, "Seed Track p_{T} Distribution (all seeds);Seed p_{T} [GeV];Seeds");
      barStyle(h, steelblue());
      drawAll({h}, true);
      auto* l = legend(0.6, 0.78, 0.94, 0.88);
      addLines(l, h, "f", {TString::Format("N=%zu", d.seedPt.size()), TString::Format("Mean: %.1f GeV", vmean(d.seedPt))});
      l->Draw();
      save(c, dir + "seed_pt.pdf");
    }
    {
      auto* c = canvas();
      auto* h = hist(d.seedEta, linspace(-3, 3, 61), kCounts, "Seed Track #eta Distribution (all seeds);Seed #eta;Seeds");
      barStyle(h, steelblue());
      drawAll({h}, false);
      save(c, dir + "seed_eta.pdf");
    }
  }
  if (!d.seedMult.empty() && vmax(d.seedMult) > 0) {
    double minNs = 1e300;
    for (double x : d.seedMult)
      if (x > 0) minNs = std::min(minNs, x);
    minNs = std::max(1.0, minNs);
    double maxNs = vmax(d.seedMult);
    Vec edges = maxNs > 1 ? logspace(std::max(0.5, minNs / 1.2), maxNs * 1.2, 51) : arange(-0.5, 5.5, 1);
    auto* c = canvas();
    gPad->SetLogx(maxNs > 1);
    auto* h = hist(d.seedMult, edges, kUnit, "Seed Multiplicity per Event;Number of seeds per event;Fraction of events");
    barStyle(h, steelblue());
    drawAll({h}, false);
    auto* l = legend(0.45, 0.82, 0.94, 0.88);
    l->AddEntry(h, TString::Format("mean=%.2e,  median=%.2e", vmean(d.seedMult), vmedian(d.seedMult)), "f");
    l->Draw();
    save(c, dir + "seed_mult_per_event.pdf");
  }
  const TString seedRule = TString::Format("#geq%d/3 muon hits, else closest #DeltaR < %g", d.seedMinHits, d.seedMaxDr);
  if (!d.effPt.empty())
    printf("Muons with a matched seed: %zu/%zu (%ld by >= %d of 3 muon hits, %ld by closest seed within Delta R < %g)\n",
           d.seedBest.size(), d.effPt.size(), d.nSeedHit, d.seedMinHits, d.nSeedDr, d.seedMaxDr);
  if (!d.seedBest.empty()) {
    Vec truth, seed, res;
    for (auto& p : d.seedBest) {
      truth.push_back(p.first);
      seed.push_back(p.second);
      res.push_back((p.second - p.first) / p.first);
    }
    double ptmax = std::max(vmax(truth), vmax(seed)) * 1.2;
    {
      auto* c = canvas();
      auto* h = hist2(truth, seed, linspace(0, ptmax, 51), linspace(0, ptmax, 51));
      drawColz(h, "Truth vs Seed p_{T} (" + seedRule + ");Truth Muon p_{T} [GeV];Best-matched Seed p_{T} [GeV]", "Counts");
      auto* diag = new TLine(0, 0, ptmax, ptmax);
      diag->SetLineColor(kRed);
      diag->SetLineStyle(2);
      diag->Draw();
      auto* l = legend(0.17, 0.82, 0.5, 0.88);
      l->AddEntry(diag, "truth = seed", "l");
      l->Draw();
      save(c, dir + "truth_vs_seed_pt_hist2d.pdf");
    }
    {
      auto* c = canvas();
      auto* h = hist(res, linspace(-2, 2, 81), kCounts,
                     "Seed p_{T} Resolution (" + seedRule + ");(p_{T}^{seed} - p_{T}^{truth}) / p_{T}^{truth};Counts");
      barStyle(h, steelblue());
      drawAll({h}, false);
      auto* l = legend(0.65, 0.78, 0.94, 0.88);
      addLines(l, h, "f", {TString::Format("Mean: %.3f", vmean(res)), TString::Format("RMS: %.3f", vstd(res))});
      l->Draw();
      save(c, dir + "seed_pt_resolution.pdf");
    }
  }

  // --- hit multiplicity of the muon-matched tracks
  Vec nhitEdges = arange(-0.5, 20.5, 1);
  if (!d.totalNr.empty()) {
    auto* c = canvas();
    auto* h = hist(d.totalNr, nhitEdges, kCounts, "Hit Multiplicity (Non-Refitted, Muon Matched);N hits;Tracks");
    barStyle(h, steelblue());
    drawAll({h}, true);
    auto* l = legend(0.65, 0.78, 0.94, 0.88);
    addLines(l, h, "f", {TString::Format("Mean: %.2f", vmean(d.totalNr)), TString::Format("Median: %.1f", vmedian(d.totalNr))});
    l->Draw();
    save(c, dir + "hit_mult_nonrefit.pdf");
  }
  if (!d.totalNr.empty() && !d.nbibNr.empty()) {
    Vec muonHits;
    for (size_t k = 0; k < d.totalNr.size() && k < d.nbibNr.size(); ++k) muonHits.push_back(d.totalNr[k] - d.nbibNr[k]);
    auto* c = canvas();
    auto* hm = hist(muonHits, nhitEdges, kCounts, "Hit Multiplicity Overlay (Non-Refitted, Muon Matched);N hits;Tracks");
    auto* hb = hist(d.nbibNr, nhitEdges);
    barStyle(hm, steelblue(), 0.6);
    barStyle(hb, salmon(), 0.6);
    drawAll({hm, hb}, true);
    auto* l = legend(0.3, 0.8, 0.94, 0.88, 0.028);
    l->AddEntry(hm, TString::Format("Muon-matched hits  Mean: %.2f, Median: %.1f", vmean(muonHits), vmedian(muonHits)), "f");
    l->AddEntry(hb, TString::Format("Non-muon hits  Mean: %.2f, Median: %.1f", vmean(d.nbibNr), vmedian(d.nbibNr)), "f");
    l->Draw();
    save(c, dir + "hit_mult_overlay_nonrefit.pdf");
  }

  // --- muon-matched track parameters
  plotTrackDist(d.recoPtNr, logspace(1, 5000, 101), "Track p_{T} (Muon Matched);p_{T} [GeV];Fraction of tracks", true, true,
                dir + "track_pt_overlay.pdf");
  plotTrackDist(d.etaNr, linspace(-3, 3, 61), "Track #eta (Muon Matched);#eta;Fraction of tracks", false, true,
                dir + "track_eta_overlay.pdf");
  plotTrackDist(d.phiNr, linspace(-M_PI, M_PI, 65), "Track #phi (Muon Matched);#phi;Fraction of tracks", false, true,
                dir + "track_phi_overlay.pdf");
  if (!d.d0Nr.empty()) {
    double m = std::max(std::fabs(vmin(d.d0Nr)), std::fabs(vmax(d.d0Nr))) * 1.1;
    if (m <= 0) m = 1;
    plotTrackDist(d.d0Nr, linspace(-m, m, 81), "Track d_{0} (Muon Matched);d_{0} [mm];Fraction of tracks", false, false,
                  dir + "track_d0_overlay.pdf");
  }
  if (!d.z0Nr.empty()) {
    double m = std::max(std::fabs(vmin(d.z0Nr)), std::fabs(vmax(d.z0Nr))) * 1.1;
    if (m <= 0) m = 1;
    plotTrackDist(d.z0Nr, linspace(-m, m, 81), "Track z_{0} (Muon Matched);z_{0} [mm];Fraction of tracks", false, false,
                  dir + "track_z0_overlay.pdf");
  }
  if (!d.nbibNr.empty()) {
    auto* c = canvas();
    auto* h = hist(d.nbibNr, arange(-0.5, std::max(vmax(d.nbibNr), 20.0) + 1.5, 1), kPerEntry,
                   "# BIB Hits per Track (Muon Matched);# non-muon (BIB) hits per track;Fraction of tracks");
    stepStyle(h, salmon());
    drawAll({h}, true);
    auto* l = legend(0.5, 0.82, 0.94, 0.88);
    l->AddEntry(h, TString::Format("SiTracks (non-refitted)  mean=%.2f", vmean(d.nbibNr)), "l");
    l->Draw();
    save(c, dir + "num_bib_hits_overlay.pdf");
  }
  if (!d.muFracNr.empty()) {
    auto* c = canvas();
    auto* h = hist(d.muFracNr, linspace(0, 1, 21), kPerEntry,
                   "Muon-Matched Hit Fraction per Track;Muon-matched hit fraction;Fraction of tracks");
    stepStyle(h, salmon());
    drawAll({h}, true);
    auto* l = legend(0.17, 0.82, 0.6, 0.88);
    l->AddEntry(h, TString::Format("SiTracks (non-refitted)  mean=%.3f", vmean(d.muFracNr)), "l");
    l->Draw();
    save(c, dir + "muonhitfrac_overlay.pdf");
  }

  // --- cellID ID verification: (system, side, layer)
  if (!d.idCounts.empty()) {
    printf("[ID DIAG] cellID decoding: (system, side, layer) -> hit count (all events)\n");
    printf("[ID DIAG] system map (EDM4hep): 1=VXDBarrel 2=VXDEndcap 3=ITBarrel 4=ITEndcap 5=OTBarrel 6=OTEndcap\n");
    std::set<int> sysIds;
    std::set<std::pair<int, int>> pairs;
    for (auto& kv : d.idCounts) {
      sysIds.insert(kv.first[0]);
      pairs.insert({kv.first[0], kv.first[2]});
    }
    TString pl = "[";
    for (auto it = pairs.begin(); it != pairs.end(); ++it)
      pl += TString::Format("%s(%d, %d)", it == pairs.begin() ? "" : ", ", it->first, it->second);
    pl += "]";
    printf("[ID DIAG] unique system IDs present: %s\n", pyList(sysIds).Data());
    printf("[ID DIAG] unique (system, layer) pairs: %s\n", pl.Data());
    TString sizes;
    for (auto& kv : d.idSample) {
      auto seen = d.idSeen.find(kv.first);
      sizes += TString::Format("%ssys=%d:%zu/%lld", sizes.IsNull() ? "" : ", ", kv.first, kv.second.size(),
                               seen == d.idSeen.end() ? 0LL : (long long)seen->second);
    }
    printf("[ID DIAG] random sample sizes per system: %s\n", sizes.Data());

    if (!d.idSample.empty()) {
      double zmin = 1e300, zmax = -1e300, rmax = 0;
      size_t nsamp = 0;
      for (auto& kv : d.idSample)
        for (auto& s : kv.second) {
          zmin = std::min(zmin, (double)s.z);
          zmax = std::max(zmax, (double)s.z);
          rmax = std::max(rmax, (double)s.r);
          ++nsamp;
        }
      double dz = 0.05 * (zmax - zmin + 1);
      auto* c = canvas(1800, 600);
      grid(c, 3, 1, TString::Format("cellID ID verification  (%zu hits, randomly sampled per subdetector)", nsamp));
      // key(sample) -> legend label, per panel
      auto panel = [&](int pad, const TString& title, std::function<int(int, const IdSample&)> key,
                       std::function<TString(int)> label) {
        c->cd(pad);
        gPad->SetGrid();
        gPad->DrawFrame(zmin - dz, 0, zmax + dz, rmax * 1.05, title + ";z [mm];r [mm]");
        std::map<int, std::pair<Vec, Vec>> groups;
        for (auto& kv : d.idSample)
          for (auto& s : kv.second) {
            auto& g = groups[key(kv.first, s)];
            g.first.push_back(s.z);
            g.second.push_back(s.r);
          }
        auto* l = legend(0.16, 0.5, 0.5, 0.88, 0.025);
        int k = 0;
        for (auto& g : groups) {
          TGraph* gr = scatter(g.second.first, g.second.second, mpl(k++), 0.5, 20, 0.3);
          l->AddEntry(gr, label(g.first), "p");
        }
        l->Draw();
      };
      panel(1, "Hits in r-z, colored by cellID `system`", [](int s, const IdSample&) { return s; },
            [](int s) { return TString::Format("sys=%d (%s)", s, relname(s)); });
      panel(2, "Hits in r-z, colored by cellID `layer`", [](int, const IdSample& x) { return x.layer; },
            [](int layer) { return TString::Format("layer=%d", layer); });
      panel(3, "Hits in r-z, colored by (system, side)", [](int s, const IdSample& x) { return s * 10 + x.side; },
            [](int combo) { return TString::Format("sys.side=%d.%d", combo / 10, combo % 10); });
      save(c, dir + "id_verification_rz.pdf");
    }
  }

  // --- z = 0 module seam: the IT/OT barrels tile their modules with an even
  // nz, so a module edge lands exactly on z = 0; the VXD barrel (odd
  // nmodules = 5) has a module centred there instead.
  if (!d.zsZ.empty()) {
    const int barrels[3] = {1, 3, 5};
    const char* titles[3] = {"VXD Barrel (odd nmodules -> module centred on z=0)",
                             "IT Barrel (even nz -> module EDGE on z=0)", "OT Barrel (even nz -> module EDGE on z=0)"};
    std::map<int, std::pair<Vec, Vec>> zoom;
    long nzoom = 0;
    double rmin = 1e300, rmax = 0;
    for (size_t k = 0; k < d.zsZ.size(); ++k)
      if (std::fabs(d.zsZ[k]) <= ZSEAM_ZOOM_HALF) {
        zoom[d.zsSys[k]].first.push_back(d.zsZ[k]);
        zoom[d.zsSys[k]].second.push_back(d.zsR[k]);
        rmin = std::min(rmin, (double)d.zsR[k]);
        rmax = std::max(rmax, (double)d.zsR[k]);
        ++nzoom;
      }
    {
      auto* c = canvas(900, 600);
      gPad->SetGrid();
      double pad = nzoom ? 0.05 * (rmax - rmin + 1) : 0;
      gPad->DrawFrame(-ZSEAM_ZOOM_HALF, nzoom ? rmin - pad : 0, ZSEAM_ZOOM_HALF, nzoom ? rmax + pad : 1600,
                      TString::Format("Merged tracker hits within #pm%g mm of z = 0  (%ld hits, %lld events);z [mm];r [mm]",
                                      ZSEAM_ZOOM_HALF, nzoom, i));
      auto* l = legend(0.16, 0.6, 0.55, 0.88, 0.025);
      int k = 0;
      for (auto& g : zoom) {
        TGraph* gr = scatter(g.second.first, g.second.second, mpl(k++), 0.6, 20, 0.4);
        l->AddEntry(gr, TString::Format("sys=%d (%s), %zu hits", g.first, relname(g.first), g.second.first.size()), "p");
      }
      if (nzoom == 0) {
        TLatex t;
        t.SetTextAlign(22);
        t.DrawLatexNDC(0.5, 0.5, TString::Format("no hits with |z| < %g mm", ZSEAM_ZOOM_HALF));
      }
      vline(0.0, kRed, 2, 2, 0.8);
      l->Draw();
      save(c, dir + "zseam_rz_zoom.pdf");
    }
    {
      auto* c = canvas(1000, 1100);
      grid(c, 1, 3, TString::Format("Hit z profile around the module seam at z = 0  (%lld events)", i));
      printf("[ZSEAM] hits within |z| < %.1f mm, and within |z| < %.1f mm, per subsystem:\n", d.zseamCollectHalf,
             ZSEAM_ZOOM_HALF);
      for (int b = 0; b < 3; ++b) {
        c->cd(b + 1);
        gPad->SetGrid();
        Vec zz;
        long nnear = 0;
        for (size_t k = 0; k < d.zsZ.size(); ++k)
          if (d.zsSys[k] == barrels[b]) {
            zz.push_back(d.zsZ[k]);
            nnear += std::fabs(d.zsZ[k]) <= ZSEAM_ZOOM_HALF;
          }
        printf("[ZSEAM]   sys=%d (%-10s): %7zu within collect window, %7ld within zoom window\n", barrels[b],
               relname(barrels[b]), zz.size(), nnear);
        int nb = (int)std::lround(2 * d.zseamCollectHalf / 0.1);
        auto* h = new TH1D(uid(), TString::Format("%s   [%zu hits, %ld with |z| < %g mm];z [mm];Hits / 0.1 mm", titles[b],
                                                  zz.size(), nnear, ZSEAM_ZOOM_HALF),
                           nb, -d.zseamCollectHalf, d.zseamCollectHalf);
        fillH(h, zz);
        stepStyle(h, steelblue(), 1, 1);
        h->Draw("HIST");
        vline(0.0, kRed, 2, 1, 0.8);
      }
      save(c, dir + "zseam_z_profile.pdf");
    }
  }

  // --- truth muon kinematics: what the gun produced, independent of the reconstruction
  if (!d.effPt.empty()) {
    auto* c = canvas(1100, 800);
    grid(c, 2, 2, TString::Format("Truth kinematics, |PDG| = %d  (%lld events)", d.signalPdg, i));
    double lo = 1e300;
    for (double x : d.effPt)
      if (x > 0) lo = std::min(lo, x);
    lo = lo < 1e300 ? std::max(lo, 1e-3) : 1e-3;
    struct P { const Vec* v; Vec edges; const char* x; const char* t; bool logx; };
    std::vector<P> panels = {{&d.effPt, logspace(lo * 0.9, vmax(d.effPt) * 1.1, 61), "Truth p_{T} [GeV]", "Truth p_{T}", true},
                             {&d.effEta, linspace(-3, 3, 61), "Truth #eta", "Truth #eta", false},
                             {&d.effPhi, linspace(-M_PI, M_PI, 61), "Truth #phi [rad]", "Truth #phi", false},
                             {&d.effTheta, linspace(0, 180, 181), "Truth #theta [deg]", "Truth #theta", false}};
    for (int k = 0; k < 4; ++k) {
      c->cd(k + 1);
      gPad->SetGrid();
      gPad->SetLogx(panels[k].logx);
      auto* h = hist(*panels[k].v, panels[k].edges, kCounts,
                     TString::Format("%s   mean=%.3g, median=%.3g, N=%zu;%s;Truth particles", panels[k].t,
                                     vmean(*panels[k].v), vmedian(*panels[k].v), panels[k].v->size(), panels[k].x));
      stepStyle(h, steelblue());
      drawAll({h}, false);
    }
    save(c, dir + "truth_kinematics.pdf");
    printf("[TRUTH] |PDG|=%d: %zu particles in %lld events; pT [%.2f, %.2f] GeV, theta [%.2f, %.2f] deg\n", d.signalPdg,
           d.effPt.size(), i, vmin(d.effPt), vmax(d.effPt), vmin(d.effTheta), vmax(d.effTheta));
  } else {
    printf("[TRUTH] WARNING: no MCParticles with |PDG| = %d found in %lld events. Every efficiency will read 0%%. "
           "Set the SIGNAL_PDG env var to the gun's species.\n", d.signalPdg, i);
  }

  // --- tracker r-z occupancy: hits per event in each (z, r) bin
  double occSum = 0;
  for (double x : d.occ) occSum += x;
  if (occSum > 0 && i > 0) {
    auto* h = new TH2D(uid(), TString::Format("Tracker r-z occupancy (hits / event, %lld events);z [mm];r [mm]", i), OCC_NZ,
                       OCC_ZMIN, OCC_ZMAX, OCC_NR, 0, OCC_RMAX);
    double mn = 1e300, mx = 0;
    for (int iz = 0; iz < OCC_NZ; ++iz)
      for (int ir = 0; ir < OCC_NR; ++ir) {
        double v = d.occ[iz * OCC_NR + ir] / double(i);
        h->SetBinContent(iz + 1, ir + 1, v);
        if (v > 0) {
          mn = std::min(mn, v);
          mx = std::max(mx, v);
        }
      }
    auto* c = canvas(1200, 600);
    gPad->SetRightMargin(0.12);
    gPad->SetLogz();
    gStyle->SetPalette(kRainBow);   // ~ matplotlib's jet; empty bins stay white
    h->SetMinimum(mn);
    h->SetMaximum(mx);
    h->Draw("COLZ");
    save(c, dir + "tracker_occupancy_rz.pdf");
  }

  // --- truth muon eta
  if (!d.effEta.empty()) {
    auto* c = canvas();
    auto* h = hist(d.effEta, linspace(-3, 3, 51), kCounts, "Truth Muon #eta Distribution;Truth Muon #eta;Muons");
    barStyle(h, steelblue());
    drawAll({h}, false);
    auto* l = legend(0.7, 0.78, 0.94, 0.88);
    addLines(l, h, "f", {TString::Format("Mean: %.2f", vmean(d.effEta)), TString::Format("Median: %.2f", vmedian(d.effEta))});
    l->Draw();
    save(c, dir + "truth_muon_eta.pdf");
  }

  // --- tracking efficiency
  if (!d.effEta.empty()) {
    auto frac = [&](const Vec& m, int region) {   // region: 0 all, 1 barrel |eta|<0.6, 2 endcap
      double n = 0, p = 0;
      for (size_t k = 0; k < m.size(); ++k) {
        bool brl = std::fabs(d.effEta[k]) < 0.6;
        if (region == 1 && !brl) continue;
        if (region == 2 && brl) continue;
        n += 1;
        p += m[k];
      }
      return n > 0 ? p / n : NaN;
    };
    TString labAll = TString::Format("AllTracks (int: %.1f%%, brl: %.1f%%, EC: %.1f%%)", 100 * frac(d.effAll, 0),
                                     100 * frac(d.effAll, 1), 100 * frac(d.effAll, 2));
    TString labSi = TString::Format("SiTracks (int: %.1f%%, brl: %.1f%%, EC: %.1f%%)", 100 * frac(d.effSi, 0),
                                    100 * frac(d.effSi, 1), 100 * frac(d.effSi, 2));
    for (int v = 0; v < 2; ++v) {
      const Vec& vals = v == 0 ? d.effEta : d.effPt;
      Vec edges = v == 0 ? linspace(-2.5, 2.5, 26) : logspace(2, 2500, 33);
      auto* c = canvas();
      gPad->SetGrid();
      gPad->SetLogx(v == 1);
      gPad->DrawFrame(edges.front(), 0, edges.back(), 1.05,
                      v == 0 ? "Tracking Efficiency vs Truth #eta;Truth Muon #eta;Tracking Efficiency"
                             : "Tracking Efficiency vs Truth p_{T};Truth Muon p_{T} [GeV];Tracking Efficiency");
      Vec x, y, ey;
      effPoints(vals, d.effAll, edges, x, y, ey);
      auto* ga = effGraph(x, y, ey, purple(), 33);
      effPoints(vals, d.effSi, edges, x, y, ey);
      auto* gs = effGraph(x, y, ey, steelblue(), 20);
      if (v == 0) {
        for (double xl : {-0.6, 0.6}) {
          auto* l = new TLine(xl, 0, xl, 1.05);
          l->SetLineColorAlpha(gray(), 0.7);
          l->SetLineStyle(2);
          l->Draw();
        }
        TLatex t;
        t.SetTextAlign(21);
        t.SetTextSize(0.025);
        t.SetTextColor(gray());
        t.DrawLatex(-1.55, 0.03, "Endcap (|#eta|>0.6)");
        t.DrawLatex(0.0, 0.03, "Barrel (|#eta|<0.6)");
        t.DrawLatex(1.55, 0.03, "Endcap (|#eta|>0.6)");
      }
      auto* l = legend(0.2, 0.14, 0.9, 0.26, 0.027);
      l->AddEntry(ga, labAll, "lp");
      l->AddEntry(gs, labSi, "lp");
      l->Draw();
      save(c, dir + (v == 0 ? "efficiency_vs_eta.pdf" : "efficiency_vs_pt.pdf"));
    }
  }

  // --- average hits per muon-matched track vs layer
  const size_t nMatched = d.totalNr.size();
  if (nMatched > 0) {
    auto* c = canvas(1100, 1200);
    grid(c, 2, 3, "Average Hits per Muon Track vs. Layer (Muon Matched)");
    for (int s = 1; s <= 6; ++s) {
      c->cd(s);
      auto it = d.layerHitsNr.find(s);
      if (it == d.layerHitsNr.end() || it->second.empty()) {
        emptyPanel(SYS_NAMES[s]);
        continue;
      }
      int lmin = it->second.begin()->first, lmax = it->second.rbegin()->first;
      auto* h = new TH1D(uid(), TString::Format("%s;Layer;Avg hits / muon track", SYS_NAMES[s]), lmax - lmin + 1,
                         lmin - 0.5, lmax + 0.5);
      for (auto& kv : it->second) h->SetBinContent(kv.first - lmin + 1, double(kv.second) / nMatched);
      barStyle(h, mpl(0), 0.8, mpl(0));
      h->SetBarWidth(0.6);
      h->SetBarOffset(0.2);
      h->SetMinimum(0);
      h->SetMaximum(h->GetMaximum() * 1.15);
      h->Draw("BAR");
      gPad->SetGridy();
      auto* l = legend(0.7, 0.8, 0.94, 0.88);
      l->AddEntry(h, "Non-Refitted", "f");
      l->Draw();
    }
    save(c, dir + "avg_hits_per_layer.pdf");
  }
}

// ---- cross-input overlays --------------------------------------------------------------------
struct Styles {
  std::vector<int> col, ls;
  std::vector<bool> variant;
  std::vector<TString> label;
};

// Inputs whose names differ only by "extended_timing_" are the same seeding
// with a wider timing window: one colour, the extended-timing one dashed
// (hatched in the bar plot).
Styles makeStyles(const std::vector<InputData>& in) {
  Styles s;
  std::vector<TString> tags, base;
  for (auto& d : in) tags.push_back(d.tag);
  std::map<TString, int> colorOf;
  for (auto& t : tags) {
    TString b = t;
    b.ReplaceAll("extended_timing_", "");
    base.push_back(b);
    if (!colorOf.count(b)) {
      int next = (int)colorOf.size();
      colorOf[b] = next;
    }
  }
  for (size_t k = 0; k < in.size(); ++k) {
    bool v = base[k] != tags[k] && std::find(tags.begin(), tags.end(), base[k]) != tags.end();
    s.variant.push_back(v);
    s.col.push_back(mpl(colorOf[base[k]]));
    s.ls.push_back(v ? 2 : 1);
    s.label.push_back(in[k].label);
  }
  return s;
}

// One step histogram per input with data; legend label via labelOf (empty = the input label).
void overlay1D(const std::vector<InputData>& in, const Styles& st, std::function<const Vec*(const InputData&)> data,
               const Vec& edges, Norm norm, const TString& title, bool logx, bool logy, const TString& path,
               std::function<TString(size_t, const Vec&)> labelOf = nullptr) {
  bool any = false;
  for (auto& d : in) any |= !data(d)->empty();
  if (!any) return;
  auto* c = canvas();
  gPad->SetLogx(logx);
  std::vector<TH1*> hs;
  auto* l = legend(0.5, 0.88 - 0.045 * in.size(), 0.94, 0.88, 0.026);
  for (size_t k = 0; k < in.size(); ++k) {
    const Vec* v = data(in[k]);
    if (v->empty()) continue;
    auto* h = hist(*v, edges, norm, title);
    stepStyle(h, st.col[k], st.ls[k]);
    hs.push_back(h);
    l->AddEntry(h, labelOf ? labelOf(k, *v) : st.label[k], "l");
  }
  drawAll(hs, logy);
  l->Draw();
  save(c, path);
}

void plotOverlays(const std::vector<InputData>& in, const TString& outdir) {
  gSystem->mkdir(outdir, kTRUE);
  const TString dir = outdir + "/";
  Styles st = makeStyles(in);
  auto has = [&](std::function<const Vec*(const InputData&)> f) {
    for (auto& d : in)
      if (!f(d)->empty()) return true;
    return false;
  };

  // 1. muon-matched track multiplicity per event
  if (has([](const InputData& d) { return &d.muonMatchedSi; })) {
    double mx = 0;
    for (auto& d : in)
      if (!d.muonMatchedSi.empty()) mx = std::max(mx, vmax(d.muonMatchedSi));
    overlay1D(in, st, [](const InputData& d) { return &d.muonMatchedSi; }, arange(-0.5, mx + 1.5, 1), kUnit,
              "Muon-Matched Track Multiplicity per Event (SiTracks, non-refitted);Number of muon-matched tracks per event;Fraction of events",
              false, true, dir + "muon_matched_track_mult_per_event_overlay.pdf");
  }

  // 2. track multiplicity per event: one panel per collection, each with its own range
  if (has([](const InputData& d) { return &d.siMult; })) {
    const char* names[3] = {"AllTracks", "DedupedTracks", "SiTracks"};
    auto c = canvas(1200, 500);
    grid(c, 3, 1, "Track Multiplicity per Event");
    for (int p = 0; p < 3; ++p) {
      auto get = [p](const InputData& d) -> const Vec& { return p == 0 ? d.allMult : (p == 1 ? d.dedupMult : d.siMult); };
      c->cd(p + 1);
      double lo = 1e300, hi = 0;
      for (auto& d : in)
        for (double x : get(d))
          if (x > 0) {
            lo = std::min(lo, x);
            hi = std::max(hi, x);
          }
      if (hi <= 0) {
        emptyPanel(names[p]);
        continue;
      }
      gPad->SetLogx();
      Vec edges = logspace(lo / 1.2, hi * 1.2, 51);
      std::vector<TH1*> hs;
      auto* l = legend(0.16, 0.88 - 0.05 * in.size(), 0.94, 0.88, 0.03);
      for (size_t k = 0; k < in.size(); ++k) {
        const Vec& v = get(in[k]);
        if (v.empty()) continue;
        auto* h = hist(v, edges, kUnit, TString::Format("%s;Track multiplicity per event;Fraction of events", names[p]));
        stepStyle(h, st.col[k], st.ls[k]);
        hs.push_back(h);
        l->AddEntry(h, TString::Format("%s  (mean=%.3g)", st.label[k].Data(), vmean(v)), "l");
      }
      drawAll(hs, false);
      l->Draw();
    }
    save(c, dir + "track_multiplicity_per_event_overlay.pdf");
  }

  // 3. chi2 reduced
  overlay1D(in, st, [](const InputData& d) { return &d.chi2Nr; }, logspace(1e-2, 1e6, 30), kPerEntry,
            "Chi2 Reduced (Non-Refitted, Muon Matched);Chi2 Reduced;Fraction of tracks", true, true,
            dir + "chi2_reduced_nonrefit_overlay.pdf");

  // 4. pixel / inner / outer / total hits
  for (int stage = 0; stage < 2; ++stage) {
    const TString tag = stage == 0 ? "Non-Refitted" : "AllTracks";
    auto sel = [stage](const InputData& d, int p) -> const Vec& {
      if (stage == 0) return p == 0 ? d.pixNr : p == 1 ? d.innerNr : p == 2 ? d.outerNr : d.totalNr;
      return p == 0 ? d.pixAll : p == 1 ? d.innerAll : p == 2 ? d.outerAll : d.totalAll;
    };
    bool any = false;
    for (auto& d : in) any |= !sel(d, 3).empty();
    if (!any) continue;
    const char* pt[4] = {"Pixel Hits", "Inner Hits", "Outer Hits", "Total Hits"};
    const int xmax[4] = {10, 5, 5, 20};
    auto* c = canvas(1200, 900);
    grid(c, 2, 2);
    for (int p = 0; p < 4; ++p) {
      c->cd(p + 1);
      Vec edges = arange(-0.5, xmax[p] + 0.5, 1);
      std::vector<TH1*> hs;
      auto* l = legend(0.35, 0.88 - 0.05 * in.size(), 0.94, 0.88, 0.03);
      for (size_t k = 0; k < in.size(); ++k) {
        const Vec& v = sel(in[k], p);
        if (v.empty()) continue;
        auto* h = hist(v, edges, kPerEntry,
                       TString::Format("%s (%s);Number of hits;Fraction of muon matched tracks", pt[p], tag.Data()));
        stepStyle(h, st.col[k], st.ls[k]);
        hs.push_back(h);
        l->AddEntry(h, TString::Format("%s  (mean=%.2f, med=%.1f)", st.label[k].Data(), vmean(v), vmedian(v)), "l");
      }
      if (hs.empty()) {
        emptyPanel(TString::Format("%s (%s)", pt[p], tag.Data()));
        continue;
      }
      drawAll(hs, false);
      l->Draw();
    }
    save(c, dir + (stage == 0 ? "all_hits_combined_nonrefit_overlay.pdf" : "all_hits_combined_alltracks_overlay.pdf"));
  }

  // 5. muon-matched vs non-muon (BIB) hits
  if (has([](const InputData& d) { return &d.totalNr; }) && has([](const InputData& d) { return &d.nbibNr; })) {
    Vec edges = arange(-0.5, 20.5, 1);
    auto* c = canvas(1200, 500);
    grid(c, 2, 1);
    std::vector<TH1*> hm, hb;
    auto* lm = legend(0.5, 0.88 - 0.05 * in.size(), 0.94, 0.88, 0.03);
    auto* lb = legend(0.5, 0.88 - 0.05 * in.size(), 0.94, 0.88, 0.03);
    for (size_t k = 0; k < in.size(); ++k) {
      const InputData& d = in[k];
      if (d.totalNr.empty() || d.nbibNr.empty()) continue;
      Vec muonHits;
      for (size_t j = 0; j < d.totalNr.size() && j < d.nbibNr.size(); ++j) muonHits.push_back(d.totalNr[j] - d.nbibNr[j]);
      auto* h1 = hist(muonHits, edges, kPerEntry, "Muon-matched hits (Non-Refitted);N hits;Fraction of tracks");
      auto* h2 = hist(d.nbibNr, edges, kPerEntry, "Non-muon (BIB) hits (Non-Refitted);N hits;Fraction of tracks");
      stepStyle(h1, st.col[k], st.ls[k]);
      stepStyle(h2, st.col[k], st.ls[k]);
      hm.push_back(h1);
      hb.push_back(h2);
      lm->AddEntry(h1, st.label[k], "l");
      lb->AddEntry(h2, st.label[k], "l");
    }
    c->cd(1);
    drawAll(hm, true);
    lm->Draw();
    c->cd(2);
    drawAll(hb, true);
    lb->Draw();
    save(c, dir + "hit_mult_muon_bib_nonrefit_overlay.pdf");
  }

  // 6-10. muon-matched track parameters
  overlay1D(in, st, [](const InputData& d) { return &d.recoPtNr; }, logspace(1, 5000, 51), kPerEntry,
            "Track p_{T} (Non-Refitted, Muon Matched);p_{T} [GeV];Fraction of tracks", true, true, dir + "track_pt_overlay.pdf");
  overlay1D(in, st, [](const InputData& d) { return &d.etaNr; }, linspace(-3, 3, 31), kPerEntry,
            "Track #eta (Non-Refitted, Muon Matched);#eta;Fraction of tracks", false, true, dir + "track_eta_overlay.pdf");
  overlay1D(in, st, [](const InputData& d) { return &d.phiNr; }, linspace(-M_PI, M_PI, 33), kPerEntry,
            "Track #phi (Non-Refitted, Muon Matched);#phi;Fraction of tracks", false, true, dir + "track_phi_overlay.pdf");
  for (int p = 0; p < 2; ++p) {
    auto get = [p](const InputData& d) { return p == 0 ? &d.d0Nr : &d.z0Nr; };
    double m = 0;
    for (auto& d : in)
      for (double x : *get(d)) m = std::max(m, std::fabs(x));
    if (m <= 0) continue;
    m *= 1.1;
    overlay1D(in, st, get, linspace(-m, m, 41), kPerEntry,
              p == 0 ? "Track d_{0} (Non-Refitted, Muon Matched);d_{0} [mm];Fraction of tracks"
                     : "Track z_{0} (Non-Refitted, Muon Matched);z_{0} [mm];Fraction of tracks",
              false, false, dir + (p == 0 ? "track_d0_overlay.pdf" : "track_z0_overlay.pdf"));
  }

  // 11. truth muon eta
  overlay1D(in, st, [](const InputData& d) { return &d.effEta; }, linspace(-3, 3, 26), kCounts,
            "Truth Muon #eta Distribution;Truth Muon #eta;Muons", false, false, dir + "truth_muon_eta_overlay.pdf");

  // 12-13. SiTracks efficiency
  if (has([](const InputData& d) { return &d.effEta; })) {
    for (int v = 0; v < 2; ++v) {
      Vec edges = v == 0 ? linspace(-2.5, 2.5, 26) : logspace(2, 2500, 33);
      auto* c = canvas();
      gPad->SetGrid();
      gPad->SetLogx(v == 1);
      gPad->DrawFrame(edges.front(), 0, edges.back(), 1.05,
                      v == 0 ? "Tracking Efficiency vs #eta (SiTracks);Truth Muon #eta;Tracking Efficiency"
                             : "Tracking Efficiency vs p_{T} (SiTracks);Truth Muon p_{T} [GeV];Tracking Efficiency");
      auto* l = legend(0.2, 0.14, 0.9, 0.14 + 0.045 * in.size(), 0.027);
      for (size_t k = 0; k < in.size(); ++k) {
        const InputData& d = in[k];
        if (d.effEta.empty()) continue;
        Vec x, y, ey;
        effPoints(v == 0 ? d.effEta : d.effPt, d.effSi, edges, x, y, ey);
        auto* g = effGraph(x, y, ey, st.col[k], 20, st.ls[k]);
        l->AddEntry(g, TString::Format("%s  (%.1f%%)", st.label[k].Data(), 100 * vmean(d.effSi)), "lp");
      }
      if (v == 0)
        for (double xl : {-0.6, 0.6}) {
          auto* ln = new TLine(xl, 0, xl, 1.05);
          ln->SetLineColorAlpha(gray(), 0.7);
          ln->SetLineStyle(2);
          ln->Draw();
        }
      l->Draw();
      save(c, dir + (v == 0 ? "efficiency_vs_eta_overlay.pdf" : "efficiency_vs_pt_overlay.pdf"));
    }
  }

  // 14. average hits per layer, grouped bars
  bool anyLayers = false;
  for (auto& d : in) anyLayers |= !d.layerHitsNr.empty();
  if (anyLayers) {
    auto* c = canvas(1300, 1200);
    grid(c, 2, 3, "Avg Hits per Muon Track vs. Layer (Non-Refitted)");
    const double bw = 0.8 / std::max<size_t>(in.size(), 1);
    for (int s = 1; s <= 6; ++s) {
      c->cd(s);
      std::set<int> layers;
      for (auto& d : in) {
        auto it = d.layerHitsNr.find(s);
        if (it != d.layerHitsNr.end())
          for (auto& kv : it->second) layers.insert(kv.first);
      }
      if (layers.empty()) {
        emptyPanel(SYS_NAMES[s]);
        continue;
      }
      int lmin = *layers.begin(), lmax = *layers.rbegin();
      std::vector<TH1*> hs;
      auto* l = legend(0.55, 0.88 - 0.045 * in.size(), 0.94, 0.88, 0.025);
      double ymax = 0;
      for (size_t k = 0; k < in.size(); ++k) {
        const InputData& d = in[k];
        auto* h = new TH1D(uid(), TString::Format("%s;Layer;Avg hits / muon track", SYS_NAMES[s]), lmax - lmin + 1,
                           lmin - 0.5, lmax + 0.5);
        auto it = d.layerHitsNr.find(s);
        size_t n = d.totalNr.size();
        if (it != d.layerHitsNr.end() && n > 0)
          for (auto& kv : it->second) h->SetBinContent(kv.first - lmin + 1, double(kv.second) / n);
        h->SetFillColorAlpha(st.col[k], 0.8);
        h->SetFillStyle(st.variant[k] ? 3354 : 1001);
        h->SetLineColor(st.col[k]);
        h->SetBarWidth(bw);
        h->SetBarOffset(0.1 + k * bw);
        ymax = std::max(ymax, h->GetMaximum());
        hs.push_back(h);
        l->AddEntry(h, st.label[k], "f");
      }
      hs[0]->SetMinimum(0);
      hs[0]->SetMaximum(ymax > 0 ? ymax * 1.15 : 1);
      for (size_t k = 0; k < hs.size(); ++k) hs[k]->Draw(k ? "BAR SAME" : "BAR");
      gPad->SetGridy();
      l->Draw();
    }
    save(c, dir + "avg_hits_per_layer_nonrefit_overlay.pdf");
  }

  // 15. seed pT
  {
    std::vector<Vec> seedPts;
    for (auto& d : in) seedPts.push_back(toVec(d.seedPt));
    overlay1D(in, st, [&](const InputData& d) { return &seedPts[&d - &in[0]]; }, logspace(1, 5000, 41), kPerEntry,
              "Seed Track p_{T} Distribution (all seeds);Seed p_{T} [GeV];Fraction of seeds", true, true,
              dir + "seed_pt_overlay.pdf");
  }

  // 16. seeds per event (events with seeds)
  {
    std::vector<Vec> pos(in.size());
    double lo = 1e300, hi = 0;
    for (size_t k = 0; k < in.size(); ++k)
      for (double x : in[k].seedMult)
        if (x > 0) {
          pos[k].push_back(x);
          lo = std::min(lo, x);
          hi = std::max(hi, x);
        }
    if (hi > 0) {
      auto* c = canvas();
      gPad->SetLogx();
      Vec edges = logspace(std::max(0.5, lo / 1.2), hi * 1.2, 26);
      std::vector<TH1*> hs;
      auto* l = legend(0.45, 0.88 - 0.09 * in.size(), 0.94, 0.88, 0.026);
      for (size_t k = 0; k < in.size(); ++k) {
        if (pos[k].empty()) continue;
        auto* h = hist(pos[k], edges, kUnit, "Seed Multiplicity per Event;Number of seeds per event;Fraction of events");
        stepStyle(h, st.col[k], st.ls[k]);
        hs.push_back(h);
        addLines(l, h, "l", {st.label[k], TString::Format("mean=%.2e,  median=%.2e", vmean(pos[k]), vmedian(pos[k]))});
      }
      drawAll(hs, false);
      l->Draw();
      save(c, dir + "seed_mult_per_event_overlay.pdf");
    }
  }

  // 17. seed pT resolution
  {
    std::vector<Vec> res(in.size());
    for (size_t k = 0; k < in.size(); ++k)
      for (auto& p : in[k].seedBest) res[k].push_back((p.second - p.first) / p.first);
    const InputData& d0 = in[0];
    overlay1D(in, st, [&](const InputData& d) { return &res[&d - &in[0]]; }, linspace(-2, 2, 41), kPerEntry,
              TString::Format("Seed p_{T} Resolution (#geq%d/3 muon hits, else closest #DeltaR < %g);"
                              "(p_{T}^{seed} - p_{T}^{truth}) / p_{T}^{truth};Fraction of seeds",
                              d0.seedMinHits, d0.seedMaxDr),
              false, false, dir + "seed_pt_resolution_overlay.pdf",
              [&](size_t k, const Vec& v) {
                return TString::Format("%s  #mu=%.3f #sigma=%.3f", st.label[k].Data(), vmean(v), vstd(v));
              });
  }

  // 18-21. residuals and muon-hit fraction (SiTracks)
  overlay1D(in, st, [](const InputData& d) { return &d.ptResNr; }, linspace(-1, 1, 41), kPerEntry,
            "p_{T} Residual (SiTracks (loose / non-refitted), Muon Matched);"
            "(p_{T}^{reco} - p_{T}^{truth}) / p_{T}^{truth};Fraction of tracks",
            false, true, dir + "pt_residual_loose_overlay.pdf");
  overlay1D(in, st, [](const InputData& d) { return &d.muFracNr; }, linspace(0, 1, 11), kPerEntry,
            "Muon-Matched Hit Fraction (SiTracks (loose / non-refitted));Muon-matched hit fraction;Fraction of tracks",
            false, true, dir + "muonhitfrac_loose_overlay.pdf",
            [&](size_t k, const Vec& v) { return TString::Format("%s  mean=%.3f", st.label[k].Data(), vmean(v)); });
  auto musig = [&](size_t k, const Vec& v) {
    return TString::Format("%s  #mu=%.3f #sigma=%.3f", st.label[k].Data(), vmean(v), vstd(v));
  };
  overlay1D(in, st, [](const InputData& d) { return &d.d0ResNr; }, linspace(-0.05, 0.05, 41), kPerEntry,
            "d_{0} Residual (SiTracks (loose / non-refitted), Muon Matched);d_{0}^{reco} - d_{0}^{truth} [mm];Fraction of tracks",
            false, true, dir + "d0_residual_loose_overlay.pdf", musig);
  overlay1D(in, st, [](const InputData& d) { return &d.z0ResNr; }, linspace(-0.05, 0.05, 41), kPerEntry,
            "z_{0} Residual (SiTracks (loose / non-refitted), Muon Matched);z_{0}^{reco} - z_{0}^{truth} [mm];Fraction of tracks",
            false, true, dir + "z0_residual_loose_overlay.pdf", musig);
}

void setStyle() {
  gROOT->SetBatch(kTRUE);
  gErrorIgnoreLevel = kWarning;   // no "pdf file ... has been created" per plot
  TH1::AddDirectory(kFALSE);
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(1);
  gStyle->SetTitleBorderSize(0);
  gStyle->SetTitleFillColor(0);
  gStyle->SetTitleStyle(0);
  gStyle->SetTitleX(0.5);
  gStyle->SetTitleAlign(23);
  gStyle->SetTitleFontSize(0.04);
  gStyle->SetPadTickX(1);
  gStyle->SetPadTickY(1);
  gStyle->SetLegendBorderSize(0);
  gStyle->SetEndErrorSize(3);
}

void run(const std::vector<TString>& inputs, const std::vector<TString>& labels, const TString& overlayDir) {
  setStyle();
  if (!labels.empty() && labels.size() != inputs.size()) {
    Error("trackPlotter", "%zu labels for %zu inputs; they must match", labels.size(), inputs.size());
    return;
  }
  std::vector<InputData> all;
  for (size_t k = 0; k < inputs.size(); ++k) {
    InputData d;
    d.dir = inputs[k];
    while (d.dir.EndsWith("/")) d.dir.Chop();
    d.tag = gSystem->BaseName(d.dir);
    d.label = labels.empty() ? d.tag : labels[k];
    if (!readInput(d)) continue;
    TString plotdir = d.tag + "_plots";
    gSystem->mkdir(plotdir, kTRUE);
    printHeader(d, plotdir);
    plotEntry(d, plotdir);
    // Keep only what the overlays use.
    for (auto* v : {&d.rawNhits, &d.rawChi2, &d.rawPt, &d.rawD0, &d.rawZ0, &d.seedEta, &d.zsZ, &d.zsR}) {
      v->clear();
      v->shrink_to_fit();
    }
    d.displays.clear();
    d.allRz.clear();
    d.idSample.clear();
    all.push_back(std::move(d));
  }
  if (all.size() > 1) plotOverlays(all, overlayDir);
}

}  // namespace tp

void trackPlotter(TString onlyDir = "") {
  const TString base = "/ospool/uc-shared/project/futurecolliders/larsonma/muon_pT2_1000GeV_theta88_92/";

  // Each input: a directory of <sample>_ntuple<N>.root files (condor/ntuple).
  std::vector<TString> inputs = {
      base + "ntuples_fullBIB_coned_VXD_InOut_TwoWay",
      base + "ntuples_fullBIB_coned_IT_OutIn",
      base + "ntuples_fullBIB_coned_OT_OutIn",
  };
  // Overlay legend labels, one per input ({} = the directory names).
  std::vector<TString> labels = {"VXD Barrel, InOut", "IT Barrel, OutIn", "OT Barrel, OutIn"};
  TString overlayDir = "VXD_IT_OT_Coned_10062026";

  //inputs = {
  //    base + "ntuples_fullBIB_uncone_VXD_InOut_TwoWay",
  //    base + "ntuples_fullBIB_uncone_IT_OutIn",
  //    base + "ntuples_fullBIB_uncone_OT_OutIn",
  //    base + "ntuples_fullBIB_uncone_extended_timing_VXD_InOut_TwoWay",
  //    base + "ntuples_fullBIB_uncone_extended_timing_IT_OutIn",
  //    base + "ntuples_fullBIB_uncone_extended_timing_OT_OutIn",
  //};
  //labels = {"VXD Barrel Seeding", "IT Barrel Seeding", "OT Barrel Seeding",
  //          "VXD Barrel Seeding, Ext. Timing", "IT Barrel Seeding, Ext. Timing", "OT Barrel Seeding, Ext. Timing"};
  //overlayDir = "overlay_plots_TrackMult_10022026_100pcBIB";

  tp::MAX_FILES_PER_INPUT = -1;   // > 0: read only the first N ntuple files of each input
  if (!onlyDir.IsNull()) {
    inputs = {onlyDir};
    labels = {};
  }
  tp::run(inputs, labels, overlayDir);
}
