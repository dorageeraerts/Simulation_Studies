//******************************************************************************
// PrimaryGeneratorAction_PartGun.cc
//
// Michael Tytgat
//******************************************************************************
//

#include "PrimaryGeneratorAction_PG.hh"

#include "G4ParticleGun.hh"
#include "G4Event.hh"
#include "G4SystemOfUnits.hh"
#include <G4MuonMinus.hh>

//----------------------------------------------------------------------------//
PrimaryGeneratorAction_PG::PrimaryGeneratorAction_PG()
{
  // define a particle gun
  particleGun = new G4ParticleGun();

  // Create the table containing all particle names
  //particleTable = G4ParticleTable::GetParticleTable();
  particleTable = G4ParticleTable::GetParticleTable();
  G4String particleName = "mu-";
  G4ParticleDefinition *particle = particleTable->FindParticle(particleName);
  particleGun->SetParticleDefinition(particle);  
}

//----------------------------------------------------------------------------//
PrimaryGeneratorAction_PG::~PrimaryGeneratorAction_PG()
{
  if (particleGun) delete particleGun;
}

//----------------------------------------------------------------------------//
void PrimaryGeneratorAction_PG::GeneratePrimaries(G4Event* anEvent)

{ 
  G4double run1   = 500. * m;
G4double run2   = 250. * m;

G4double width  = run1 + run2;
  G4double toeX  = -width / 2.;
  G4double world_sizeX = 760.*m;
  G4double world_sizeZ  = 180*m;
  
  G4double detYZ = 1.0*m;
  G4double detX  = 1.7*cm;
  G4double gap_x = 1.7*m;
  G4double angle1 = 10. * deg;

G4double x1 = -0.5*world_sizeX + 10.0*m;
G4double x2 = x1 + gap_x;

G4double z2 = -0.5*world_sizeZ + (x2 - toeX)*std::tan(angle1) + 0.51*detYZ;
  G4double x = x2+1*m; 
  //G4double y = 0; 
  G4double y = 0;
  //G4double z = 0;
  G4double z = z2;

  G4ThreeVector pos(x, y, z); 

  G4double px = -1.; 
  G4double py = 0; 
  G4double pz = 0;

  G4ThreeVector mom(px, py, pz); 

  particleGun->SetParticlePosition(pos);
  particleGun->SetParticleMomentumDirection(mom.unit());
  particleGun->SetParticleEnergy(4. * GeV);    

  particleGun->GeneratePrimaryVertex(anEvent);
}

std::string PrimaryGeneratorAction_PG::GetInfoSummary() const {
    std::ostringstream oss;
    oss << "Particle: " << particleGun->GetParticleDefinition()->GetParticleName()
        << ", Energy: " << particleGun->GetParticleEnergy()/CLHEP::MeV << " MeV"
        << ", Direction: " << particleGun->GetParticleMomentumDirection();
    return oss.str();
}