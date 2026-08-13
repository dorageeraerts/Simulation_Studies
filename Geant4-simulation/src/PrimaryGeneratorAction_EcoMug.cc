/// \file PrimaryGeneratorAction_EcoMug.cc
/// \brief Implementation of the PrimaryGeneratorAction_EcoMug class

#include "PrimaryGeneratorAction_EcoMug.hh"
//#include "GlobalParameters.hh"

#include "G4LogicalVolumeStore.hh"
#include "G4LogicalVolume.hh"
#include "G4Box.hh"
#include "G4RunManager.hh"
#include "G4Run.hh"
#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4ParticleDefinition.hh"
#include "G4SystemOfUnits.hh"
#include "Randomize.hh"
#include "G4Geantino.hh"
#include "G4IonTable.hh"
#include "G4Event.hh"
#include "G4PhysicalConstants.hh"
#include <fstream>
#include <cmath>
#include "EcoMug.h"
#include "G4ChargedGeantino.hh"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <chrono>
#include "G4AnalysisManager.hh"
//#include "MuonEventInfo.hh"
#include "G4TransportationManager.hh"
#include "EcomugMessenger.hh"
//#include "TFile.h"
//#include "TH2D.h"
//#include "HistogramManager.hh"

using namespace std;
using namespace chrono;
#define DEG_TO_RAD(deg) ((deg) * M_PI / 180.0)
EMLog::TLogLevel EMLog::ReportingLevel = WARNING;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

PrimaryGeneratorAction_EcoMug::PrimaryGeneratorAction_EcoMug(const std::string& inputFile): 
G4VUserPrimaryGeneratorAction(), fParticleGun(0), mu_plus(0), mu_minus(0),
  fMinTheta(0.), fMaxTheta(M_PI/2), fMinPhi(0.), fMaxPhi(2*M_PI), fMinMom(0.01), fMaxMom(1000.),
  fMinPosTheta(0.), fMaxPosTheta(M_PI/2), fMinPosPhi(0.), fMaxPosPhi(2*M_PI), genHSphere(true), customFlux(true), fHorizontalRate(138*(EMUnits::hertz/EMUnits::m2)),
  fHSphereRadius(150*cm), fHSphereCenter({56.45*cm,0*cm,-48.5*cm}), fSkyCenter({0.,0.,0.}), fSkySize({1.*m,1.*m}), seedEcomug(-1)
{ 
	const double DEG_TO_RAD = M_PI / 180.0;

	auto messenger = new EcoMugMessenger(this);

	if (!inputFile.empty()) {
        ReadConfigFile(inputFile);
    }

    if (seedEcomug == -1) {
	seedEcomug = std::chrono::system_clock::now().time_since_epoch().count();
    }

	G4int n_particle = 1;
	fParticleGun  = new G4ParticleGun(n_particle);

	mu_plus = G4ParticleTable::GetParticleTable()->FindParticle("mu+");
	mu_minus = G4ParticleTable::GetParticleTable()->FindParticle("mu-");
  }

PrimaryGeneratorAction_EcoMug::~PrimaryGeneratorAction_EcoMug()
{  
    delete fParticleGun;
    delete fGenSuite;
}

void PrimaryGeneratorAction_EcoMug::Initialize() {
    // -------- Set muon generation surface (sky or hemisphere) --------
    if (genHSphere) {
        fGenHSphere.SetUseHSphere();

        fGenHSphere.SetHSphereRadius(fHSphereRadius);
        fGenHSphere.SetHSphereCenterPosition(fHSphereCenter);
    } else {
        fGenHSphere.SetUseSky();
		fGenHSphere.SetSkySize(fSkySize);
		fGenHSphere.SetSkyCenterPosition(fSkyCenter);
    }

	fGenHSphere.SetSeed(seedEcomug);

    // -------- Set limitations on generated muon direction -------- 
    fGenHSphere.SetMinimumTheta(fMinTheta);
    fGenHSphere.SetMaximumTheta(fMaxTheta);
    //fGenHSphere.SetMinimumPhi(fMinPhi);
    //fGenHSphere.SetMaximumPhi(fMaxPhi);

    // -------- Set limitations on generation position on hemisphere --------
    fGenHSphere.SetHSphereMinPositionTheta(fMinPosTheta);
	fGenHSphere.SetHSphereMaxPositionTheta(fMaxPosTheta);
    fGenHSphere.SetHSphereMinPositionPhi(fMinPosPhi);
    fGenHSphere.SetHSphereMaxPositionPhi(fMaxPosPhi);

    // -------- Set limitations on generated muon momentum -------- 
    fGenHSphere.SetMinimumMomentum(fMinMom);
    fGenHSphere.SetMaximumMomentum(fMaxMom);

    fGenHSphere.SetHorizontalRate(fHorizontalRate * EMUnits::hertz/EMUnits::m2);

    // -------- Set custom parameterisation for cosmic muon flux --------
    if (customFlux) {
        fGenHSphere.SetDifferentialFlux([this](double p, double theta) {
            return this->J(p, theta);
        });
    }

    if (fUseBackground && !fBckPID.empty()) {
    std::vector<EcoMug> bckGenerators;
    bckGenerators.reserve(fBckPID.size());

    for (std::size_t i = 0; i < fBckPID.size(); ++i) {
        // Clone the signal generator's geometry/cuts via the copy constructor
        EcoMug bck(fGenHSphere);
        // Give the background its own flux parameterisation.
        // Swap this out per-species if you need different J's per background.
        bck.SetDifferentialFlux([this](double p, double theta) {
            return this->Jbck(p, theta);
        });
        bckGenerators.push_back(bck);
    }

    fGenSuite = new EMMultiGen(fGenHSphere, bckGenerators);
    fGenSuite->SetBckWeights(fBckWeights);
    fGenSuite->SetBckPID(fBckPID);

    G4cout << "EMMultiGen configured with " << fBckPID.size()
           << " background species." << G4endl;
}

}

// Parameterise Guan's flux (EcoMug uses m-2 sr-1 s-1 GeV-1)
G4double PrimaryGeneratorAction_EcoMug::J(G4double p, G4double theta) 
{
	G4double P1 = 0.102573;
	G4double P2 = -0.068287;
	G4double P3 = 0.958633;
	G4double P4 = 0.0407253;
	G4double P5 = 0.817285;

	G4double cos_theta = cos(theta);
	G4double num = pow(cos_theta,2) + pow(P1,2) + P2*pow(cos_theta,P3) + P4*pow(cos_theta,P5);
	G4double denom = 1 + pow(P1,2) + P2 + P4;
	G4double cos_theta_atm = sqrt(num/denom);

	// Guan's flux formula is give in terms of energy
	G4double E_muon;
	E_muon = sqrt(p*p+0.10566*0.10566);

    G4double A = 1400; 
	G4double B = E_muon*(1 + 3.64/(E_muon*pow(cos_theta_atm,1.29)));
    G4double C = 1. / (1. + 1.1*E_muon*cos_theta_atm/115.);
    G4double D = 0.054 / (1. + 1.1*E_muon*cos_theta_atm/850.);
    return A*pow(B,-2.7)*(C+D);
};

void PrimaryGeneratorAction_EcoMug::GeneratePrimaries(G4Event* anEvent)
{
    G4double muon_phi, muon_theta, muon_ptot;
    std::array<double,3> muon_pos;
    G4int pdg;

    if (fMaxPhi < 2*M_PI || fMinPhi > 0.)
    {
        do {
            pdg = GenerateOneEvent(muon_pos, muon_ptot, muon_theta, muon_phi);

            double phi0_min = 0.0;
            double phi0_max = fMaxPhi;
            double phi0_min_wrapped = 2*M_PI - fMaxPhi;
            double phi_pi_min = M_PI - fMaxPhi;
            double phi_pi_max = M_PI + fMaxPhi;

            bool in_window0 = (muon_phi >= phi0_min && muon_phi <= phi0_max) ||
                               (muon_phi >= phi0_min_wrapped && muon_phi < 2*M_PI);
            bool in_window_pi = (muon_phi >= phi_pi_min && muon_phi <= phi_pi_max);

            if (in_window0 || in_window_pi) break;
        } while (true);
    }
    else {
        pdg = GenerateOneEvent(muon_pos, muon_ptot, muon_theta, muon_phi);
    }

    G4ParticleDefinition* particle = G4ParticleTable::GetParticleTable()->FindParticle(pdg);
    if (!particle) {
        G4cerr << "PrimaryGeneratorAction_EcoMug: unknown PDG code " << pdg
               << ", defaulting to mu-" << G4endl;
        particle = mu_minus;
    }
    fParticleGun->SetParticleDefinition(particle);

    G4double a = sin(muon_theta)*cos(muon_phi);
    G4double b = sin(muon_theta)*sin(muon_phi);
    G4double c = cos(muon_theta);

    fParticleGun->SetParticlePosition(G4ThreeVector(muon_pos[0], muon_pos[1], muon_pos[2]));
    fParticleGun->SetParticleMomentumDirection(G4ParticleMomentum(a, b, c));
    fParticleGun->SetParticleMomentum(muon_ptot * GeV); // single conversion, correct mass via `particle`

    fParticleGun->GeneratePrimaryVertex(anEvent);
}

std::string PrimaryGeneratorAction_EcoMug::GetInfoSummary() const {
    fGenHSphere.GetAverageGenRateAndError(rateHSphere, errorHSphere);
	genSurfaceArea = fGenHSphere.GetGenSurfaceArea();

    std::ostringstream oss;
    oss << "Generator: " << GetGeneratorName() <<
    "\nGenerator seed: " << seedEcomug <<
	"\nCustom flux (bool): " << customFlux <<
    "\nSet horizontal rate (m-2 s-1): " << fGenHSphere.GetHorizontalRate() / (EMUnits::hertz/EMUnits::m2) << "\n";
    if (genHSphere) {
        oss << "\nGeneration surface: hemisphere" << 
        "\nfHSphereCenter (cm): ("
        << fHSphereCenter[0] / cm << ", "
        << fHSphereCenter[1] / cm << ", "
        << fHSphereCenter[2] / cm << ")\n"
        << "fHSphereRadius (cm): " << fHSphereRadius / cm << "\n\n";}
    else {
        oss << "\nGeneration surface: flat sky" << 
        "\nfSkyCenter (cm): ("
        << fSkyCenter[0] / cm << ", "
        << fSkyCenter[1] / cm << ", "
        << fSkyCenter[2] / cm << ")\n"
    << "\nfSkySize (cm x cm): ("
        << fSkySize[0] / cm << " x "
        << fSkySize[1] / cm << "\n\n";}
    
		oss << "Theta range (rad): [" 
		<< fGenHSphere.GetMinimumTheta() << ", "
		<< fGenHSphere.GetMaximumTheta() << "]\n"
		<< "Phi range (rad): [" 
		<< fGenHSphere.GetMinimumPhi() << ", "
		<< fGenHSphere.GetMaximumPhi() << "]\n"
		<< "\nThetaPos range (rad): [" 
		<< fMinPosTheta << ", "
		<< fMaxPosTheta << "]\n"
		<< "PhiPos range (rad): [" 
		<< fMinPosPhi << ", "
		<< fMaxPosPhi << "]\n"
		<< "\nMomentum range (GeV): [" 
		<< fGenHSphere.GetMinimumMomentum() << ", "
		<< fGenHSphere.GetMaximumMomentum() << "]\n"
		<< "\nMuon generation rate (m-2 s-1): " << rateHSphere
		<< "\nMuon generation rate error: " << errorHSphere
		<< "\nGeneration surface area (m^2): " << genSurfaceArea / m2
		;
    return oss.str();
}

void PrimaryGeneratorAction_EcoMug::ReadConfigFile(const std::string& filename) {
	G4cout << "DEBUG ReadConfigFile this=" << this << G4endl;
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        G4cerr << "Cannot open EcoMug input file: " << filename << G4endl;
        return;
    }
	
	std::string line;
    while (std::getline(infile, line)) {
    if (line.empty() || line[0] == '#') continue; // skip blank lines and comments
    std::istringstream iss(line);
    std::string key;
    if (!(iss >> key)) continue; // skip malformed lines

    if (key == "seed") {
        long seedVal;
        if (iss >> seedVal) { seedEcomug = seedVal; }
    if (key == "use_background") {
        double val;
        if (iss >> val) fUseBackground = static_cast<bool>(val);
    }
    else if (key == "bck_species") {
        int pid; double w;
        if (iss >> pid >> w) {
            fBckPID.push_back(pid);
            fBckWeights.push_back(w);
        }
    }
    } else {
        double val;
        if (!(iss >> val)) continue; 
        if      (key == "theta_min")        { fMinTheta = val; }
        else if (key == "theta_max")        { fMaxTheta = val; }
        else if (key == "phi_min")          { fMinPhi = val; }
        else if (key == "phi_max")          { fMaxPhi = val; }
        else if (key == "pos_theta_min")    { fMinPosTheta = val; }
        else if (key == "pos_theta_max")    { fMaxPosTheta = val; }
        else if (key == "pos_phi_min")      { fMinPosPhi = val; }
        else if (key == "pos_phi_max")      { fMaxPosPhi = val; }
        else if (key == "horizontal_rate")  { fHorizontalRate = val; }
        else if (key == "gen_hsphere")      { genHSphere = static_cast<bool>(val); }
        else if (key == "custom_flux")      { customFlux = static_cast<bool>(val); }
        else if (key == "hsphere_center_x") { fHSphereCenter[0] = val * cm; }
        else if (key == "hsphere_center_y") { fHSphereCenter[1] = val * cm; }
        else if (key == "hsphere_center_z") { fHSphereCenter[2] = val * cm; }
        else if (key == "hsphere_radius")   { fHSphereRadius = val * cm; }
        else if (key == "sky_size_x")       { fSkySize[0] = val * cm; }
        else if (key == "sky_size_y")       { fSkySize[1] = val * cm; }
        else if (key == "sky_center_x")     { fSkyCenter[0] = val * cm; }
        else if (key == "sky_center_y")     { fSkyCenter[1] = val * cm; }
        else if (key == "sky_center_z")     { fSkyCenter[2] = val * cm; }
        else if (key == "min_momentum")     { fMinMom = val * GeV; }
        else if (key == "max_momentum")     { fMaxMom = val * GeV; }
    }
}

    infile.close();
    G4cout << "EcoMug configuration loaded from " << filename << G4endl;
}

G4int PrimaryGeneratorAction_EcoMug::GenerateOneEvent(
    std::array<double,3>& pos, G4double& ptot, G4double& theta, G4double& phi)
{
    if (fUseBackground && fGenSuite) {
        fGenSuite->Generate();
        pos   = fGenSuite->GetGenerationPosition();
        ptot  = fGenSuite->GetGenerationMomentum(); // GeV/c
        theta = fGenSuite->GetGenerationTheta();
        phi   = fGenSuite->GetGenerationPhi();
        return fGenSuite->GetPID();                 // signed PDG code
    } else {
        fGenHSphere.Generate();
        pos   = fGenHSphere.GetGenerationPosition();
        ptot  = fGenHSphere.GetGenerationMomentum();
        theta = fGenHSphere.GetGenerationTheta();
        phi   = fGenHSphere.GetGenerationPhi();
        return (fGenHSphere.GetCharge() > 0) ? 13 : -13; // fold into mu+/mu- PDG
    }
}

G4double PrimaryGeneratorAction_EcoMug::Jbck(G4double p, G4double theta)
{
    return this->J(p, theta); // or a genuinely different background parameterisation
}