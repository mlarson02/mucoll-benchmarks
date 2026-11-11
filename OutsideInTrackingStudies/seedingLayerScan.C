// Outside-in seeding layers vs |eta| for MAIA_v0.
// Tracks are straight lines from the IP in r-z (a helix projects to one; low-pT
// curvature ignored). At each eta the nOuter outermost crossed sensitive layers are
// kept; their union over |eta| < etaMax is the seeding selection, printed as a
// Tracking/seeding_presets.py entry. nOuter = 3 is the bare triplet; nOuter = 4
// leaves one spare layer everywhere, so a single missing hit still leaves a triplet.
// Geometry: detector-simulation/geometries/MAIA_v0 Vertex/InnerTracker/OuterTracker XMLs
// (disk r = innermost ring inner edge to outermost ring outer edge).
// Hits: mean MergedTrackerHits per event, nominal timing, from plotLayerHits.C.
// Run: root -l -b -q 'seedingLayerScan.C(2.44)'      (3 outermost, seedingLayerScan_*.png)
//      root -l -b -q 'seedingLayerScan.C(2.44, 4)'   (4 outermost, seedingLayerScan_outer4_*.png)

#include <algorithm>
#include <cmath>
#include <vector>

struct SeedLayer {
  TString name;
  int sys, id;            // cellID system:<sys>,layer:<id>
  double r0, r1, z0, z1;  // sensitive extent [mm]: barrel r0 = r1, z0 = 0; disk z0 = z1
  double khits;           // mean hits per event [k], both endcaps summed
};

SeedLayer barrel(TString name, int sys, int id, double r, double halfZ, double khits) {
  return {name, sys, id, r, r, 0, halfZ, khits};
}
SeedLayer disk(TString name, int sys, int id, double z, double rmin, double rmax, double khits) {
  return {name, sys, id, rmin, rmax, z, z, khits};
}

// eta window in which a line from the IP crosses the layer
double etaLo(const SeedLayer& l) { return std::asinh(l.z0 / l.r1); }
double etaHi(const SeedLayer& l) { return std::asinh(l.z1 / l.r0); }
// crossing radius when crossed; orders layers by distance from the IP
double rAt(const SeedLayer& l, double eta) { return std::min(l.r1, l.z1 / std::sinh(eta)); }

typedef std::vector<std::pair<double, double>> Ranges;

// append [a, b], merging with the previous range when contiguous
void addRange(Ranges& v, double a, double b) {
  if (!v.empty() && v.back().second == a) v.back().second = b;
  else v.push_back({a, b});
}

TString fmt(const Ranges& v, const char* empty = "-") {
  if (v.empty()) return empty;
  TString s;
  for (auto& p : v) s += Form("%s%.3f-%.3f", s.IsNull() ? "" : " ", p.first, p.second);
  return s;
}

void seedingLayerScan(double etaMax = 2.44, int nOuter = 3) {
  // output names: unchanged for 3, "_outer<N>" otherwise
  const TString sfx = nOuter == 3 ? TString("") : TString::Format("_outer%d", nOuter);
  const std::vector<SeedLayer> L = {
    // VXD barrel, |z| < 65 (layers 0/1 are a doublet)
    barrel("VXD B0", 1, 0,  30.1, 65, 118.8), barrel("VXD B1", 1, 1,  32.2, 65,  92.9),
    barrel("VXD B2", 1, 2,  51.1, 65,  56.1), barrel("VXD B4", 1, 4,  74.1, 65,  42.5),
    barrel("VXD B6", 1, 6, 102.1, 65,  29.4),
    // VXD endcap doublets, r up to 112
    disk("VXD E0", 2, 0,  80.0, 25, 112, 114.2), disk("VXD E1", 2, 1,  84.3, 25, 112, 130.3),
    disk("VXD E2", 2, 2, 120.0, 31, 112, 184.7), disk("VXD E3", 2, 3, 124.3, 31, 112, 198.0),
    disk("VXD E4", 2, 4, 200.0, 38, 112, 218.7), disk("VXD E5", 2, 5, 204.3, 38, 112, 227.7),
    disk("VXD E6", 2, 6, 280.0, 53, 112,  89.2), disk("VXD E7", 2, 7, 284.3, 53, 112,  89.2),
    // IT barrel
    barrel("IT B0", 3, 0, 127, 481.6, 1572.2), barrel("IT B1", 3, 1, 340, 481.6, 796.0),
    barrel("IT B2", 3, 2, 554, 692.3,  691.4),
    // IT endcap
    disk("IT E0", 4, 0,  524,  95, 427.2, 476.5), disk("IT E1", 4, 1,  808, 147, 554.7, 224.1),
    disk("IT E2", 4, 2, 1093, 190, 552.4, 143.4), disk("IT E3", 4, 3, 1377, 212, 559.3,  79.9),
    disk("IT E4", 4, 4, 1661, 237, 554.1,  62.7), disk("IT E5", 4, 5, 1946, 264, 550.9,  39.5),
    disk("IT E6", 4, 6, 2190, 284, 555.8,  33.6),
    // OT barrel, |z| < 1264.2
    barrel("OT B0", 5, 0,  819, 1264.2, 830.7), barrel("OT B1", 5, 1, 1153, 1264.2, 547.1),
    barrel("OT B2", 5, 2, 1486, 1264.2, 352.3),
    // OT endcap, r 617.5-1430.1
    disk("OT E0", 6, 0, 1310, 617.5, 1430.1, 355.8), disk("OT E1", 6, 1, 1617, 617.5, 1430.1, 295.2),
    disk("OT E2", 6, 2, 1883, 617.5, 1430.1, 185.1), disk("OT E3", 6, 3, 2190, 617.5, 1430.1, 176.2),
  };
  const int n = L.size();
  auto crosses = [&](int i, double eta) { return etaLo(L[i]) < eta && eta < etaHi(L[i]); };

  // elementary eta intervals: no layer starts or stops inside one
  std::vector<double> edges = {0., etaMax};
  for (auto& l : L)
    for (double e : {etaLo(l), etaHi(l)})
      if (e > 0 && e < etaMax) edges.push_back(e);
  std::sort(edges.begin(), edges.end());
  edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

  // nOuter outermost crossed layers per interval, merging neighbours with the same set
  struct Seg { double a, b; std::vector<int> trip; };
  std::vector<Seg> segs;
  for (size_t k = 0; k + 1 < edges.size(); ++k) {
    const double a = edges[k], b = edges[k + 1], m = 0.5 * (a + b);
    std::vector<int> c;
    for (int i = 0; i < n; ++i) if (crosses(i, m)) c.push_back(i);
    std::sort(c.begin(), c.end(), [&](int i, int j) { return rAt(L[i], m) > rAt(L[j], m); });
    c.resize(std::min<size_t>(c.size(), nOuter));
    if (!segs.empty() && segs.back().trip == c) segs.back().b = b;
    else segs.push_back({a, b, c});
  }

  // selection = union of the outermost sets
  std::vector<bool> sel(n, false);
  std::vector<Ranges> inTrip(n);
  for (auto& s : segs)
    for (int i : s.trip) { sel[i] = true; addRange(inTrip[i], s.a, s.b); }

  // eta ranges where fewer than nOuter layers of a selection are crossed
  auto holes = [&](const std::vector<bool>& use) {
    Ranges h;
    for (size_t k = 0; k + 1 < edges.size(); ++k) {
      const double m = 0.5 * (edges[k] + edges[k + 1]);
      int nx = 0;
      for (int i = 0; i < n; ++i) nx += use[i] && crosses(i, m);
      if (nx < nOuter) addRange(h, edges[k], edges[k + 1]);
    }
    return h;
  };

  printf("\nOutermost %d crossed layers, straight tracks from the IP, |eta| < %.2f\n", nOuter, etaMax);
  printf("  %-15s  %s\n", "|eta|", "outer -> inner");
  for (auto& s : segs) {
    printf("  %.3f - %.3f    ", s.a, s.b);
    for (int i : s.trip) printf("%-8s", L[i].name.Data());
    if ((int)s.trip.size() < nOuter) printf("<- fewer than %d", nOuter);
    printf("\n");
  }

  printf("\n%-7s %-16s %8s %-4s %-12s %s\n", "layer", "cellID", "hits[k]", "sel", "crossed",
         Form("in outermost %d", nOuter));
  for (int i = 0; i < n; ++i) {
    TString tag = "";
    if (sel[i]) {
      std::vector<bool> without = sel;
      without[i] = false;
      tag = holes(without).empty() ? "yes*" : "yes";
    }
    printf("%-7s %-16s %8.1f %-4s %-12s %s\n", L[i].name.Data(), Form("system:%d,layer:%d", L[i].sys, L[i].id),
           L[i].khits, tag.Data(), fmt(Ranges(1, {etaLo(L[i]), etaHi(L[i])})).Data(), fmt(inTrip[i]).Data());
  }
  printf("  * droppable: the rest of the selection still crosses >= %d layers everywhere\n", nOuter);

  // minimal config: drop selected layers one at a time while no nOuter-layer hole opens
  std::vector<bool> minSel = sel;
  for (int i = 0; i < n; ++i) {
    if (!minSel[i]) continue;
    minSel[i] = false;
    if (!holes(minSel).empty()) minSel[i] = true;
  }

  // seeding_presets.py entry: whole system when every layer of it is selected
  auto presetOf = [&](const std::vector<bool>& use) {
    TString p;
    for (int s = 1; s <= 6; ++s) {
      TString ids;
      int nSys = 0, nUse = 0;
      for (int i = 0; i < n; ++i) {
        if (L[i].sys != s) continue;
        ++nSys;
        if (!use[i]) continue;
        ++nUse;
        if (!ids.IsNull()) ids += "|";
        ids += L[i].id;
      }
      if (nUse == 0) continue;
      if (!p.IsNull()) p += ", ";
      p += nUse == nSys ? Form("\"system:%d\"", s) : Form("\"system:%d,layer:%s\"", s, ids.Data());
    }
    return p;
  };
  auto nLayers = [&](const std::vector<bool>& use) { return (int)std::count(use.begin(), use.end(), true); };
  auto khits = [&](const std::vector<bool>& use) {
    double k = 0;
    for (int i = 0; i < n; ++i) if (use[i]) k += L[i].khits;
    return k;
  };

  // seed space-point extent along the outermost sets (extremes sit at interval edges)
  double rMax = 0, zMax = 0, dRMin = 1e9, dRMax = 0;
  for (auto& s : segs) {
    if ((int)s.trip.size() < nOuter) continue;
    for (double e : {s.a, s.b}) {
      std::vector<double> r(nOuter);
      for (int j = 0; j < nOuter; ++j) r[j] = rAt(L[s.trip[j]], e);
      rMax = std::max(rMax, r[0]);
      zMax = std::max(zMax, r[0] * std::sinh(e));
      for (int j = 0; j + 1 < nOuter; ++j) {
        dRMin = std::min(dRMin, r[j] - r[j + 1]);
        dRMax = std::max(dRMax, r[j] - r[j + 1]);
      }
    }
  }

  std::vector<bool> otOnly(n), itot(n);
  for (int i = 0; i < n; ++i) {
    otOnly[i] = L[i].sys >= 5;
    itot[i] = L[i].sys >= 3;
  }

  printf("\nSelection: %d layers, %.0fk hits/evt (all IT+OT layers: %.0fk)\n", nLayers(sel), khits(sel), khits(itot));
  printf("Minimal:   %d layers, %.0fk hits/evt (droppable layers removed)\n", nLayers(minSel), khits(minSel));
  printf("Seed SPs: r <= %.0f mm, |z| <= %.0f mm, consecutive dR %.0f-%.0f mm\n", rMax, zMax, dRMin, dRMax);
  printf("%d-layer holes, selection:      %s\n", nOuter, fmt(holes(sel), "none").Data());
  printf("%d-layer holes, minimal:        %s\n", nOuter, fmt(holes(minSel), "none").Data());
  printf("%d-layer holes, OT only (5, 6): %s\n", nOuter, fmt(holes(otOnly), "none").Data());
  const TString pn = nOuter == 3 ? TString("Outermost") : TString::Format("Outermost%d", nOuter);
  printf("\nseeding_presets.py:\n    \"%s\": [%s],\n    \"%sMinimal\": [%s],\n\n",
         pn.Data(), presetOf(sel).Data(), pn.Data(), presetOf(minSel).Data());

  // ---- plots ----
  gStyle->SetOptStat(0);
  auto col = [](int sys) { return TColor::GetColor(sys <= 2 ? "#2e8b3e" : sys <= 4 ? "#2166ac" : "#e08a2c"); };

  // eta coverage map into the current pad: grey = crossed, colour = one of the
  // nOuter outermost. xAxis = false leaves the x axis to a pad drawn below it.
  auto drawCoverage = [&](TString tag, bool xAxis, double rightMargin) {
    gPad->SetLeftMargin(0.11);
    gPad->SetRightMargin(rightMargin);
    gPad->SetTopMargin(0.10);
    gPad->SetBottomMargin(xAxis ? 0.09 : 0.015);
    TH2D* fr = new TH2D("fr" + tag, xAxis ? ";|#eta|;" : ";;", 10, 0, etaMax, n, 0, n);
    for (int i = 0; i < n; ++i) fr->GetYaxis()->SetBinLabel(i + 1, L[i].name);
    fr->GetYaxis()->SetLabelSize(0.028);
    if (!xAxis) fr->GetXaxis()->SetLabelSize(0);
    fr->Draw("AXIS");

    TBox* bCross = new TBox();
    bCross->SetFillColor(kGray);
    for (int i = 0; i < n; ++i) {
      const double hi = std::min(etaHi(L[i]), etaMax);
      if (etaLo(L[i]) < hi) bCross->DrawBox(etaLo(L[i]), i + 0.3, hi, i + 0.7);
    }
    TBox b;
    for (auto& s : segs)
      for (int i : s.trip) {
        b.SetFillColor(col(L[i].sys));
        b.DrawBox(s.a, i + 0.15, s.b, i + 0.85);
      }
    TLine sep;
    sep.SetLineStyle(3);
    sep.SetLineColor(kGray + 1);
    for (size_t k = 1; k < segs.size(); ++k) sep.DrawLine(segs[k].a, 0, segs[k].a, n);
    sep.SetLineStyle(2);
    for (int i = 1; i < n; ++i)
      if ((L[i].sys + 1) / 2 != (L[i - 1].sys + 1) / 2) sep.DrawLine(0, i, etaMax, i);

    TLatex t;
    t.SetNDC();
    t.SetTextSize(0.03);
    t.DrawLatex(0.11, 0.93, "#it{MAIA}_v0, tracks from the IP");
    TLegend* leg = new TLegend(0.55, 0.905, 0.96, 0.995);
    leg->SetNColumns(2);
    leg->SetBorderSize(0);
    leg->SetFillStyle(0);
    leg->SetTextSize(0.025);
    leg->AddEntry(bCross, "crossed", "f");
    const char* sub[3] = {"VXD", "IT", "OT"};
    for (int s = 0; s < 3; ++s) {
      TBox* lb = new TBox();
      lb->SetFillColor(col(2 * s + 1));
      leg->AddEntry(lb, Form("outermost %d (%s)", nOuter, sub[s]), "f");
    }
    leg->Draw();
  };

  TCanvas* c1 = new TCanvas("c1", "", 900, 800);
  drawCoverage("", true, 0.04);
  c1->SaveAs("seedingLayerScan" + sfx + "_eta.png");
  c1->SaveAs("seedingLayerScan" + sfx + "_eta.pdf");

  // UltraTight track efficiency, pT 0.5-5000 GeV, signed eta bins of 0.05 from -2.6
  // (maia_detector_paper/10TeV_paper/finerfineretaprintout.txt, "bins of 0.05" Histogram 1;
  // its bins 51-104 got printed further down, after the fit output)
  // const double effW = 0.05;
  // const int nHalf = 52;  // bins per eta sign
  // const double effPass[104] = {0, 0, 0, 0, 22, 50, 48, 56, 57, 58, 24, 86, 120,
  //                              60, 89, 85, 117, 125, 120, 74, 96, 91, 107, 132, 134, 44,
  //                              100, 111, 112, 122, 178, 227, 277, 131, 160, 174, 280, 322, 191,
  //                              173, 378, 404, 436, 421, 451, 419, 458, 496, 498, 468, 515, 546,
  //                              472, 506, 487, 472, 490, 439, 493, 449, 438, 369, 428, 340, 140,
  //                              239, 300, 271, 169, 108, 141, 229, 216, 139, 103, 85, 139, 97,
  //                              77, 143, 117, 137, 98, 97, 74, 106, 111, 102, 77, 73, 64,
  //                              88, 77, 41, 69, 64, 49, 34, 32, 4, 0, 0, 0, 0};
  // const double effTot[104] = {76, 86, 86, 86, 110, 104, 91, 104, 117, 119, 112, 138, 164,
  //                             142, 140, 141, 172, 190, 196, 220, 229, 219, 212, 250, 242, 278,
  //                             286, 302, 293, 333, 371, 334, 381, 347, 355, 388, 425, 411, 466,
  //                             431, 514, 494, 519, 507, 539, 479, 551, 555, 549, 529, 574, 610,
  //                             532, 568, 558, 545, 544, 518, 569, 521, 532, 448, 502, 443, 477,
  //                             401, 418, 418, 413, 368, 383, 375, 338, 344, 321, 258, 310, 282,
  //                             275, 237, 212, 243, 235, 222, 185, 187, 189, 165, 154, 185, 143,
  //                             136, 134, 125, 147, 143, 119, 102, 96, 101, 94, 75, 68, 71};

  // SiTracks efficiency, full IT+OT seeding (ITOTDisks OutIn, two-way, u/v fix, inflate 100),
  // NoBIB muons pT 0-50 GeV, signed eta bins of 0.05 from -2.45 (seed_vs_d0cuts_opt.txt)
  const double effW = 0.05;
  const int nHalf = 49;  // bins per eta sign
  const double effPass[98] = {5, 14, 35, 35, 34, 47, 44, 37, 41, 54, 59, 38, 55,
                              57, 59, 62, 57, 65, 73, 53, 72, 66, 76, 92, 91, 99,
                              101, 104, 107, 125, 95, 119, 134, 152, 139, 157, 129, 127, 141,
                              125, 157, 167, 148, 165, 161, 200, 160, 164, 202, 185, 174, 182,
                              159, 183, 170, 170, 172, 152, 151, 130, 137, 131, 160, 146, 161,
                              148, 128, 131, 112, 102, 101, 111, 102, 88, 89, 96, 82, 83,
                              76, 55, 66, 66, 62, 56, 48, 58, 41, 46, 47, 49, 39,
                              38, 42, 38, 35, 19, 28, 12};
  const double effTot[98] = {16, 23, 35, 35, 37, 49, 47, 40, 44, 55, 60, 38, 55,
                             58, 61, 65, 58, 65, 76, 56, 77, 74, 78, 97, 94, 105,
                             103, 108, 110, 131, 97, 125, 143, 163, 146, 164, 135, 136, 148,
                             131, 166, 173, 157, 170, 166, 207, 168, 169, 206, 192, 181, 187,
                             164, 188, 178, 180, 180, 159, 158, 133, 141, 137, 168, 149, 168,
                             155, 132, 134, 117, 104, 103, 118, 109, 94, 91, 98, 85, 88,
                             79, 56, 67, 68, 63, 61, 53, 59, 42, 47, 48, 50, 39,
                             42, 43, 38, 35, 19, 35, 35};
  // folded to |eta| (+eta and -eta summed), Clopper-Pearson 68% errors
  TGraphAsymmErrors* gEff = new TGraphAsymmErrors();
  for (int j = 0; j < nHalf && effW * (j + 1) <= etaMax + 1e-9; ++j) {
    const double p = effPass[nHalf + j] + effPass[nHalf - 1 - j],
                 tot = effTot[nHalf + j] + effTot[nHalf - 1 - j], e = p / tot;
    const int k = gEff->GetN();
    gEff->SetPoint(k, effW * (j + 0.5), e);
    gEff->SetPointError(k, effW / 2, effW / 2, e - TEfficiency::ClopperPearson(tot, p, 0.682689, false),
                        TEfficiency::ClopperPearson(tot, p, 0.682689, true) - e);
  }

  // coverage map with the efficiency below it on a shared |eta| axis; the wider
  // right margin holds the layer-count axis
  const double rm = 0.10;
  TCanvas* c3 = new TCanvas("c3", "", 900, 1100);
  TPad* pTop = new TPad("pTop", "", 0, 0.28, 1, 1);
  TPad* pBot = new TPad("pBot", "", 0, 0, 1, 0.28);
  pTop->Draw();
  pBot->Draw();
  pTop->cd();
  drawCoverage("_eff", false, rm);

  pBot->cd();
  pBot->SetLeftMargin(0.11);
  pBot->SetRightMargin(rm);
  pBot->SetTopMargin(0.03);
  pBot->SetBottomMargin(0.30);
  TH1D* fe = new TH1D("fe", ";;", 10, 0, etaMax);
  fe->SetMinimum(0);
  fe->SetMaximum(1.05);
  // pixel fonts, so sizes do not scale with the short pad
  for (TAxis* ax : {fe->GetXaxis(), fe->GetYaxis()}) {
    ax->SetLabelFont(43);
    ax->SetLabelSize(22);
  }
  fe->GetXaxis()->SetTickLength(0.08);
  fe->GetYaxis()->SetNdivisions(505);
  fe->Draw("AXIS");
  // axis titles placed directly: title offsets for pixel fonts in a short pad are fiddly
  TLatex at;
  at.SetNDC();
  at.SetTextFont(43);
  at.SetTextSize(26);
  at.SetTextAlign(31);
  at.DrawLatex(1 - rm, 0.05, "|#eta|");
  at.SetTextAngle(90);
  at.SetTextAlign(23);
  at.DrawLatex(0.02, 0.635, "Efficiency");

  TLine sep;
  sep.SetLineStyle(3);
  sep.SetLineColor(kGray + 1);
  for (size_t k = 1; k < segs.size(); ++k) sep.DrawLine(segs[k].a, 0, segs[k].a, 1.05);

  // layers crossed (all of them, doublets counted twice), averaged over |eta|
  // within each efficiency bin; drawn scaled onto the efficiency axis, read off
  // the right-hand axis
  const int nb = gEff->GetN();
  std::vector<double> nMean(nb, 0.);
  for (size_t k = 0; k + 1 < edges.size(); ++k) {
    const double a = edges[k], b = edges[k + 1];
    int nx = 0;
    for (int i = 0; i < n; ++i) nx += crosses(i, 0.5 * (a + b));
    for (int j = 0; j < nb; ++j) {
      const double ov = std::min(b, effW * (j + 1)) - std::max(a, effW * j);
      if (ov > 0) nMean[j] += nx * ov / effW;
    }
  }
  const double nLo = 5, nHi = 20;  // right-hand axis range
  auto toEff = [&](double v) { return (v - nLo) / (nHi - nLo) * 1.05; };
  const int nCol = TColor::GetColor("#c0392b");
  TGraph* gN = new TGraph();
  for (int j = 0; j < nb; ++j) {
    gN->SetPoint(gN->GetN(), effW * j, toEff(nMean[j]));
    gN->SetPoint(gN->GetN(), effW * (j + 1), toEff(nMean[j]));
  }
  gN->SetLineColor(nCol);
  gN->SetLineWidth(2);
  gN->Draw("L");
  TGaxis* axN = new TGaxis(etaMax, 0, etaMax, 1.05, nLo, nHi, 503, "+L");
  axN->SetLineColor(nCol);
  axN->SetLabelColor(nCol);
  axN->SetLabelFont(43);
  axN->SetLabelSize(22);
  axN->Draw();
  at.SetTextColor(nCol);
  at.SetTextAlign(21);
  at.DrawLatex(0.98, 0.635, "Layers crossed");

  gEff->SetMarkerStyle(20);
  gEff->SetMarkerSize(0.9);
  gEff->Draw("P");

  TLatex el;
  el.SetNDC();
  el.SetTextFont(43);
  el.SetTextSize(22);
  // el.SetTextAlign(33);
  // el.DrawLatex(1 - rm - 0.01, 0.92, "Efficiency: UltraTight tracks, p_{T} #in [0.5, 5000] GeV");
  // bottom left of the frame, below the layers-crossed line
  el.SetTextAlign(11);
  el.DrawLatex(pBot->GetLeftMargin() + 0.01, pBot->GetBottomMargin() + 0.04,
               "Efficiency: SiTracks, full IT+OT seeding, p_{T} #in [0, 50] GeV");
  c3->SaveAs("seedingLayerScan" + sfx + "_eta_efficiency.png");
  c3->SaveAs("seedingLayerScan" + sfx + "_eta_efficiency.pdf");

  // r-z view: layers of a selection thick, the rest grey
  auto drawRZ = [&](const std::vector<bool>& use, TString tag, TString title) {
    const double zFr = 2400, rFr = 1700;
    TCanvas* c = new TCanvas("c_rz" + tag, "", 1000, 750);
    c->SetLeftMargin(0.10);
    c->SetRightMargin(0.10);
    c->SetTopMargin(0.07);
    c->SetBottomMargin(0.11);
    TH2D* frz = new TH2D("frz" + tag, ";|z| [mm];r [mm]", 10, 0, zFr, 10, 0, rFr);
    frz->Draw("AXIS");

    TLine g;
    g.SetLineStyle(2);
    g.SetLineColor(kGray + 1);
    TLatex gl;
    gl.SetTextSize(0.025);
    gl.SetTextColor(kGray + 2);
    for (double e : {0.5, 1.0, 1.5, 2.0, etaMax}) {
      const double sh = std::sinh(e), r = std::min(rFr, zFr / sh);
      g.DrawLine(0, 0, r * sh, r);
      gl.SetTextAlign(r < rFr ? 12 : 21);
      gl.DrawLatex(r * sh, r, Form(" #eta = %.2f", e));
    }

    TLine ln;
    TLatex nm;
    nm.SetTextSize(0.022);
    for (int i = 0; i < n; ++i) {
      ln.SetLineColor(use[i] ? col(L[i].sys) : kGray + 1);
      ln.SetLineWidth(use[i] ? 6 : 2);
      ln.DrawLine(L[i].z0, L[i].r0, L[i].z1, L[i].r1);
      if (!use[i]) continue;
      nm.SetTextColor(col(L[i].sys));
      nm.SetTextAlign(L[i].z0 == L[i].z1 ? 21 : 12);  // disk: above, barrel: right of the end
      nm.DrawLatex(L[i].z1 + (L[i].z0 == L[i].z1 ? 0 : 15), L[i].r1 + (L[i].z0 == L[i].z1 ? 15 : 0), L[i].name);
    }

    TLatex t2;
    t2.SetNDC();
    t2.SetTextSize(0.035);
    t2.DrawLatex(0.10, 0.945, Form("#it{MAIA}_v0: %s, |#eta| < %.2f", title.Data(), etaMax));
    c->SaveAs("seedingLayerScan" + sfx + "_rz" + tag + ".png");
    c->SaveAs("seedingLayerScan" + sfx + "_rz" + tag + ".pdf");
  };
  drawRZ(sel, "", Form("outermost-%d seeding layers", nOuter));
  drawRZ(minSel, "_minimalConfig", Form("minimal outermost-%d seeding layers", nOuter));
}
