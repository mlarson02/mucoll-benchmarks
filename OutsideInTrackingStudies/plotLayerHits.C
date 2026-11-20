// Average total hits per tracking layer, nominal timing (MAIA-style layer plot).
// Numbers: per-event means of MergedTrackerHits from seed_track_mult_comparison.txt
// (endcap layers count both sides).
// Also makes the hits/cm^2 occupancy plot (plotOccupancy, below) from the same numbers,
// plus a copy with the Perlmutter BIB occupancy overlaid.
// Run: root -l -b -q plotLayerHits.C
//
// perl = the same per-event means for Jackson's Perlmutter BIB digi
// (JacksonBIBDigi/digi_output_full_bib_100_0.edm4hep.root, 50 events): the six
// {VXD,IT,OT}{Barrel,Endcap}Hits collections MergedTrackerHits merges, cellID decoded
// as in edm4hepTrackAnalyzer.py (system = id & 0x1F, layer = (id >> 7) & 0x3F).
//
// area = sensitive area per layer [cm^2] from Mira's study_occupancy.py layer_area
// (per side there; endcaps here are 2x since the hits count both sides). Her VXD barrel
// has MuColl_v1 doublets 0-7; MAIA keeps only layers 0, 1, 2, 4, 6.

struct LayerHits { TString sys; int layer; double mean; double perl; double area; };
void plotOccupancy(const std::vector<LayerHits>& d, bool overlay);

void plotLayerHits() {
  std::vector<LayerHits> d = {
    //  sys        layer   mean        perl       area
    {"VXD Barrel",  0,  118840.22,  29903.10,    270.40},
    {"VXD Barrel",  1,   92852.24,  23398.16,    270.40},
    {"VXD Barrel",  2,   56130.07,  14519.96,    448.50},
    {"VXD Barrel",  4,   42470.05,  11086.48,    655.20},
    {"VXD Barrel",  6,   29449.61,   7542.36,    904.80},
    {"VXD Endcaps", 0,  114184.34,  29026.00,    778.00},
    {"VXD Endcaps", 1,  130304.34,  33180.74,    778.00},
    {"VXD Endcaps", 2,  184652.17,  46920.94,    757.92},
    {"VXD Endcaps", 3,  198031.83,  50233.34,    757.92},
    {"VXD Endcaps", 4,  218701.24,  55990.56,    728.72},
    {"VXD Endcaps", 5,  227710.78,  58377.68,    728.72},
    {"VXD Endcaps", 6,   89168.56,  22934.14,    624.96},
    {"VXD Endcaps", 7,   89204.34,  22942.68,    624.96},
    {"IT Barrel",   0, 1572172.07, 404224.84,   8117.85},
    {"IT Barrel",   1,  795968.83, 204252.12,  22034.16},
    {"IT Barrel",   2,  691350.63, 177527.52,  51678.81},
    {"IT Endcaps",  0,  476451.63, 122194.00,  13279.30},
    {"IT Endcaps",  1,  224089.49,  57402.30,  21223.17},
    {"IT Endcaps",  2,  143412.61,  36815.80,  20156.08},
    {"IT Endcaps",  3,   79861.54,  20385.68,  19800.39},
    {"IT Endcaps",  4,   62694.63,  16030.28,  18614.74},
    {"IT Endcaps",  5,   39529.15,  10087.00,  17191.95},
    {"IT Endcaps",  6,   33552.59,   8613.58,  16599.13},
    {"OT Barrel",   0,  830651.49, 213914.12, 140032.91},
    {"OT Barrel",   1,  547104.44, 141442.42, 194828.39},
    {"OT Barrel",   2,  352255.46,  90444.98, 249623.88},
    {"OT Endcaps",  0,  355772.00,  91661.96, 139090.91},
    {"OT Endcaps",  1,  295243.17,  76034.10, 139090.91},
    {"OT Endcaps",  2,  185054.24,  47437.94, 139090.91},
    {"OT Endcaps",  3,  176211.78,  45105.28, 139090.91},
  };
  const int n = d.size();
  const double ymin = 1e4, ymax = 1e7;

  gStyle->SetOptStat(0);
  TCanvas* c = new TCanvas("c", "", 900, 600);
  c->SetLogy();
  c->SetTicky(1);  // y ticks on the right too
  c->SetLeftMargin(0.12);
  c->SetRightMargin(0.04);
  c->SetTopMargin(0.07);
  c->SetBottomMargin(0.12);

  TH1D* h = new TH1D("h", ";Tracking Detector Layer;Average Total Hits per Layer", n, 0, n);
  for (int i = 0; i < n; ++i) {
    h->SetBinContent(i + 1, d[i].mean);
    h->GetXaxis()->SetBinLabel(i + 1, Form("%d", d[i].layer));
  }
  const int col = TColor::GetColor("#CCCC00");
  h->SetFillColor(col);
  h->SetLineColor(col);
  h->SetMinimum(ymin);
  h->SetMaximum(ymax);
  h->GetXaxis()->SetLabelSize(0.035);
  h->GetXaxis()->SetTitleOffset(1.2);
  h->GetYaxis()->SetTitleOffset(1.2);
  h->Draw("HIST");

  // dashed boundary + rotated red label at the start of each subdetector
  TLine line;
  line.SetLineStyle(2);
  line.SetLineWidth(3);
  line.SetLineColor(kGray + 1);
  TLatex sec;
  sec.SetTextAngle(90);
  sec.SetTextAlign(12);
  sec.SetTextSize(0.035);
  sec.SetTextColor(kRed);
  for (int i = 0; i < n; ++i) {
    if (i > 0 && d[i].sys == d[i - 1].sys) continue;
    if (i > 0) line.DrawLine(i, ymin, i, ymax);
    sec.DrawLatex(i + 0.5, 1.3 * ymin, d[i].sys);
  }

  TLatex t;
  t.SetNDC();
  t.SetTextFont(42);  // regular; #bf only on "Muon Collider"
  t.SetTextSize(0.04);
  t.SetTextAlign(31);
  t.DrawLatex(0.96, 0.945, "#it{MAIA} Detector Concept");
  t.SetTextAlign(11);
  t.DrawLatex(0.16, 0.86, "#bf{#it{Muon Collider}}   #sqrt{s} = 10 TeV");
  t.DrawLatex(0.16, 0.80, "Simulation, with 100% BIB");

  TLegend* leg = new TLegend(0.70, 0.80, 0.94, 0.88);
  leg->SetBorderSize(0);
  leg->SetFillStyle(0);
  leg->SetTextSize(0.04);
  leg->AddEntry(h, "Nominal timing (v9)", "f");
  leg->Draw();

  c->SaveAs("layerHits_nominalTiming.png");
  c->SaveAs("layerHits_nominalTiming.pdf");

  plotOccupancy(d, false);
  plotOccupancy(d, true);
}

// Average hits/cm^2 per tracking layer (MAIA paper occupancy-plot style): mean / area.
// overlay: also draw perl / area as a line ("Perlmutter BIB"), saved as *_perlmutterBIB.
void plotOccupancy(const std::vector<LayerHits>& d, bool overlay) {
  const int n = d.size();
  const TString tag = overlay ? "_perlmutterBIB" : "";
  const double ymin = 0.3, ymax = 1e5;  // MAIA paper range is 1-1e5
  auto yAt = [&](double f) { return ymin * std::pow(ymax / ymin, f); };  // log-axis fraction

  gStyle->SetOptStat(0);
  TCanvas* c = new TCanvas("cOcc" + tag, "", 900, 600);
  c->SetLogy();
  c->SetTicky(1);  // y ticks on the right too
  c->SetLeftMargin(0.12);
  c->SetRightMargin(0.04);
  c->SetTopMargin(0.07);
  c->SetBottomMargin(0.12);

  TH1D* h = new TH1D("hOcc" + tag, ";Tracking Detector Layer;Average number of hits / cm^{2}", n, 0, n);
  for (int i = 0; i < n; ++i) h->SetBinContent(i + 1, d[i].mean / d[i].area);
  const int col = TColor::GetColor("#CCCC00");
  h->SetFillColor(col);
  h->SetLineColor(col);
  h->SetMinimum(ymin);
  h->SetMaximum(ymax);
  h->GetXaxis()->SetLabelSize(0);
  h->GetXaxis()->SetTickLength(0);
  h->GetXaxis()->SetTitleOffset(1.2);
  h->GetYaxis()->SetTitleOffset(1.2);
  h->Draw("HIST");
  TH1D* hp = nullptr;
  if (overlay) {
    hp = new TH1D("hOccPerl", "", n, 0, n);
    for (int i = 0; i < n; ++i) hp->SetBinContent(i + 1, d[i].perl / d[i].area);
    hp->SetLineColor(kBlue + 1);
    hp->SetLineWidth(3);
    hp->Draw("HIST SAME");
  }
  gPad->RedrawAxis();

  // dashed boundary + rotated red label on the first layer of each subdetector
  TLine line;
  line.SetLineStyle(2);
  line.SetLineWidth(3);
  line.SetLineColor(kGray + 1);
  TLatex sec;
  sec.SetTextAngle(90);
  sec.SetTextAlign(12);
  sec.SetTextSize(0.035);
  sec.SetTextColor(kRed);
  for (int i = 0; i < n; ++i) {
    if (i > 0 && d[i].sys == d[i - 1].sys) continue;
    if (i > 0) line.DrawLine(i, ymin, i, yAt(0.6));
    sec.DrawLatex(i + 0.5, yAt(0.36), d[i].sys);
  }

  TLatex t;
  t.SetNDC();
  t.SetTextFont(42);  // regular; #bf only on "Muon Collider"
  t.SetTextSize(0.04);
  t.SetTextAlign(31);
  t.DrawLatex(0.96, 0.945, "#it{MAIA} Detector Concept");
  t.SetTextAlign(11);
  t.DrawLatex(0.17, 0.86, "#bf{#it{Muon Collider}}   #sqrt{s} = 10 TeV");
  t.DrawLatex(0.17, 0.80, "Simulation, with 100% BIB");

  TLegend* leg = new TLegend(0.68, overlay ? 0.72 : 0.78, 0.94, 0.88);
  leg->SetBorderSize(0);
  leg->SetFillStyle(0);
  leg->SetTextSize(0.04);
  leg->AddEntry(h, "Nominal timing (v9)", "f");
  if (overlay) leg->AddEntry(hp, "Perlmutter BIB", "l");
  leg->Draw();

  c->SaveAs("occupancy_nominalTiming" + tag + ".png");
  c->SaveAs("occupancy_nominalTiming" + tag + ".pdf");
}
