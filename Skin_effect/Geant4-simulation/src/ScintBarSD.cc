/// \file ScintBarSD.cc
/// \brief Implementation of the ScintBarSD class

#include "ScintBarSD.hh"

#include "ScintBarHit.hh"

#include "G4AffineTransform.hh"
#include "G4HCofThisEvent.hh"
#include "G4SDManager.hh"
#include "G4Step.hh"
#include "G4StepPoint.hh"
#include "G4VPhysicalVolume.hh"
#include "G4VTouchable.hh"
#include "G4SystemOfUnits.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

ScintbarSD::ScintbarSD(G4String name) : G4VSensitiveDetector(name), fHitsCollection(0), fHCID(-1)
{
  collectionName.insert("ScintbarsColl");
}

ScintbarSD::~ScintbarSD() {}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void ScintbarSD::Initialize(G4HCofThisEvent* hce)
{
  fHitsCollection = new ScintbarHitsCollection(SensitiveDetectorName, collectionName[0]);
  if (fHCID < 0) {
    fHCID = G4SDManager::GetSDMpointer()->GetCollectionID(fHitsCollection);
  }
  hce->AddHitsCollection(fHCID, fHitsCollection);
  fHitCount = 0;
  fEntryPointMap.clear();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4bool ScintbarSD::ProcessHits(G4Step* step, G4TouchableHistory* history)
{
  auto track = step->GetTrack();
  auto prestep  = step->GetPreStepPoint();
  auto poststep = step->GetPostStepPoint();
  auto touchable = prestep->GetTouchable();

  G4double edep    = step->GetTotalEnergyDeposit();
  G4double stepLen = step->GetStepLength();

  if (edep == 0. && stepLen == 0.) return false;

  // Panel1 -> copy number 0, Panel2 -> copy number 1 (see DetectorConstruction)
  G4int PanelID = touchable->GetVolume()->GetCopyNo();
  //G4cout << "PanelID" << PanelID << G4endl;
  G4int trackID = track->GetTrackID();

  // Search for existing hit for this track-panel combination in this event
  // (one hit per muon per panel, energy summed across all its steps in that panel)
  ScintbarHit* hit = nullptr;
  for (size_t i = 0; i < fHitsCollection->entries(); i++) {
    auto* h = (*fHitsCollection)[i];
    if (h->GetTrackID()  == trackID &&
        h->GetPanelID()  == PanelID) {
      hit = h;
      break;
    }
  }

  if (!hit) { // no entry yet for this track-panel combination
    if (edep == 0.) {
      // No energy deposited yet, cache geometric entry point but don't
      // create the hit yet (memory efficiency, same as before)
      fEntryPointMap[{trackID, PanelID}] = prestep->GetPosition();
      return false;
    }

    hit = new ScintbarHit();
    hit->SetPanelID(PanelID);
    G4cout << "Creating hit for track " << trackID << " in panel " << PanelID << G4endl;
    hit->SetTrackID(trackID);
    hit->SetPDGcode(track->GetDefinition()->GetPDGEncoding());
    hit->SetParentId(track->GetParentID());
    hit->SetHitTime(prestep->GetGlobalTime());

    auto key = std::make_pair(trackID, PanelID);
    auto it  = fEntryPointMap.find(key);
    if (it != fEntryPointMap.end()) {
      hit->SetEntryPoint(it->second);
      fEntryPointMap.erase(it);
    } else {
      hit->SetEntryPoint(prestep->GetPosition());
    }

    fHitsCollection->insert(hit);
  }

  hit->AddEdep(edep);

  if (stepLen > 0.) {
    hit->AddPathLength(stepLen);
    hit->SetExitPoint(poststep->GetPosition());
  }
  return true;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void ScintbarSD::EndOfEvent( G4HCofThisEvent *hitCollection ) {}

