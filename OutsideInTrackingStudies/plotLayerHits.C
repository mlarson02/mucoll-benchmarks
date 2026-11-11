// Average total hits per tracking layer, nominal timing (MAIA-style layer plot).
// Numbers: per-event means of MergedTrackerHits from seed_track_mult_comparison.txt
// (endcap layers count both sides).
// Also makes the hits/cm^2 occupancy plot (plotOccupancy, below) from the same numbers.
// Run: root -l -b -q plotLayerHits.C
//
// area = sensitive area per layer [cm^2] from Mira's study_occupancy.py layer_area
// (per side there; endcaps here are 2x since the hits count both sides). Her VXD barrel
// has MuColl_v1 doublets 0-7; MAIA keeps only layers 0, 1, 2, 4, 6.

struct LayerHits { TString sys; int layer; double mean; double area; };
void plotOccupancy(const std::vector<LayerHits>& d);

void plotLayerHits() {
  std::vector<LayerHits> d = {
    {"VXD Barrel",  0,  118840.22,    270.40},
    {"VXD Barrel",  1,   92852.24,    270.40},
    {"VXD Barrel",  2,   56130.07,    448.50},
    {"VXD Barrel",  4,   42470.05,    655.20},
    {"VXD Barrel",  6,   29449.61,    904.80},
    {"VXD Endcaps", 0,  114184.34,    778.00},
    {"VXD Endcaps", 1,  130304.34,    778.00},
    {"VXD Endcaps", 2,  184652.17,    757.92},
    {"VXD Endcaps", 3,  198031.83,    757.92},
    {"VXD Endcaps", 4,  218701.24,    728.72},
    {"VXD Endcaps", 5,  227710.78,    728.72},
    {"VXD Endcaps", 6,   89168.56,    624.96},
    {"VXD Endcaps", 7,   89204.34,    624.96},
    {"IT Barrel",   0, 1572172.07,   8117.85},
    {"IT Barrel",   1,  795968.83,  22034.16},
    {"IT Barrel",   2,  691350.63,  51678.81},
    {"IT Endcaps",  0,  476451.63,  13279.30},
    {"IT Endcaps",  1,  224089.49,  21223.17},
    {"IT Endcaps",  2,  143412.61,  20156.08},
    {"IT Endcaps",  3,   79861.54,  19800.39},
    {"IT Endcaps",  4,   62694.63,  18614.74},
    {"IT Endcaps",  5,   39529.15,  17191.95},
    {"IT Endcaps",  6,   33552.59,  16599.13},
    {"OT Barrel",   0,  830651.49, 140032.91},
    {"OT Barrel",   1,  547104.44, 194828.39},
    {"OT Barrel",   2,  352255.46, 249623.88},
    {"OT Endcaps",  0,  355772.00, 139090.91},
    {"OT Endcaps",  1,  295243.17, 139090.91},
    {"OT Endcaps",  2,  185054.24, 139090.91},
    {"OT Endcaps",  3,  176211.78, 139090.91},
  };
  const int n = d.size();
  const double ymin = 1e4, ymax = 1e7;

  gStyle->SetOptStat(0);
  TCanvas* c = new TCanvas("c", "", 900, 600);
  c->SetLogy();
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
  t.DrawLatex(0.16, 0.80, "Simulation, with v9 BIB+IPP");

  TLegend* leg = new TLegend(0.70, 0.80, 0.94, 0.88);
  leg->SetBorderSize(0);
  leg->SetFillStyle(0);
  leg->SetTextSize(0.04);
  leg->AddEntry(h, "Nominal timing", "f");
  leg->Draw();

  c->SaveAs("layerHits_nominalTiming.png");
  c->SaveAs("layerHits_nominalTiming.pdf");

  plotOccupancy(d);
}

// Average hits/cm^2 per tracking layer (MAIA paper occupancy-plot style): mean / area.
void plotOccupancy(const std::vector<LayerHits>& d) {
  const int n = d.size();
  const double ymin = 1, ymax = 1e5;  // MAIA paper range
  auto yAt = [&](double f) { return ymin * std::pow(ymax / ymin, f); };  // log-axis fraction

  gStyle->SetOptStat(0);
  TCanvas* c = new TCanvas("cOcc", "", 900, 600);
  c->SetLogy();
  c->SetLeftMargin(0.12);
  c->SetRightMargin(0.04);
  c->SetTopMargin(0.07);
  c->SetBottomMargin(0.12);

  TH1D* h = new TH1D("hOcc", ";Tracking Detector Layer;Average number of hits / cm^{2}", n, 0, n);
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
  t.DrawLatex(0.17, 0.80, "Simulation, with v9 BIB+IPP");

  TLegend* leg = new TLegend(0.68, 0.78, 0.94, 0.88);
  leg->SetBorderSize(0);
  leg->SetFillStyle(0);
  leg->SetTextSize(0.04);
  leg->AddEntry(h, "Nominal timing", "f");
  leg->Draw();

  c->SaveAs("occupancy_nominalTiming.png");
  c->SaveAs("occupancy_nominalTiming.pdf");
}
