/// \file EventAction.cc
/// \brief Implementation of the EventAction class

#include "EventAction.hh"

#include "TrackingAction.hh"
#include "ScintBarHit.hh"

#include "G4Event.hh"
#include "G4RunManager.hh"
#include "G4EventManager.hh"
#include "G4HCofThisEvent.hh"
#include "G4VHitsCollection.hh"
#include "G4SDManager.hh"
#include "G4SystemOfUnits.hh"
#include "G4ios.hh"
#include "G4AnalysisManager.hh"
//#include "SteppingAction.hh"
#include <sys/resource.h>

using std::array;
using std::vector;

// Utility function which finds a hit collection with the given Id
// and print warnings if not found
G4VHitsCollection* GetHC(const G4Event* event, G4int collId)
{
  /*auto hce = event->GetHCofThisEvent();
  if (!hce) {
    G4ExceptionDescription msg;
    msg << "No hits collection of this event found." << G4endl;
    G4Exception("EventAction::EndOfEventAction()", "Code001", JustWarning, msg);
    return nullptr;
  }

  auto hc = hce->GetHC(collId);
  if (!hc) {
    G4ExceptionDescription msg;
    msg << "Hits collection " << collId << " of this event not found." << G4endl;
    G4Exception("EventAction::EndOfEventAction()", "Code001", JustWarning, msg);
  }
  return hc;*/
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

EventAction::EventAction()
: G4UserEventAction(),
    fScintbarsHCID(-1)
{
  // set printing per each event
  G4RunManager::GetRunManager()->SetPrintProgress(1);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void EventAction::BeginOfEventAction(const G4Event* event)
{
  G4int eventID = event->GetEventID();
    if (eventID % 50000 == 0) 
        G4cout << ">>> Starting event " << eventID << "..." << G4endl;

  // Find hit collections by names (just once)
  if ( fScintbarsHCID == -1 ) {
    auto SDManager = G4SDManager::GetSDMpointer();
    fScintbarsHCID = SDManager->GetCollectionID("Scintbars/ScintbarsColl");
  }
  
  TrackingAction::Instance()->ResetParents();
  //fTouchedRock=false;
  //if (fSteppingAction) fSteppingAction->Reset();
  fScatteredEvent = false;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void EventAction::EndOfEventAction(const G4Event*)
{
    if(!fScatteredEvent)
        return;

    G4double pIn = fIncomingMomentum.mag();

    G4double zenithIn = fIncomingMomentum.theta();

    G4double azimuthIn = fIncomingMomentum.phi();

    G4double pOut = fOutgoingMomentum.mag();

    G4double zenithOut = fOutgoingMomentum.theta();

    G4double azimuthOut = fOutgoingMomentum.phi();

    G4double scatteringAngle = fIncomingMomentum.angle(fOutgoingMomentum);


    G4cout << "========== Muon scattering ==========" << G4endl;

    G4cout << "pIn = "
           << pIn/MeV << " MeV"
           << G4endl;

    G4cout << "zenithIn = "
           << zenithIn/deg << " deg"
           << G4endl;

    G4cout << "azimuthIn = "
           << azimuthIn/deg << " deg"
           << G4endl;

    G4cout << "pOut = "
           << pOut/MeV << " MeV"
           << G4endl;

    G4cout << "zenithOut = "
           << zenithOut/deg << " deg"
           << G4endl;

    G4cout << "azimuthOut = "
           << azimuthOut/deg << " deg"
           << G4endl;

    G4cout << "scatteringAngle = "
           << scatteringAngle/deg << " deg"
           << G4endl;

    G4cout << "====================================="
           << G4endl;


    auto analysisManager = G4AnalysisManager::Instance();

    analysisManager->FillNtupleDColumn(0,pIn);
    analysisManager->FillNtupleDColumn(1,zenithIn);
    analysisManager->FillNtupleDColumn(2,azimuthIn);
    analysisManager->FillNtupleDColumn(3,pOut);
    analysisManager->FillNtupleDColumn(4,zenithOut);
    analysisManager->FillNtupleDColumn(5,azimuthOut);
    analysisManager->FillNtupleDColumn(6,scatteringAngle);

    analysisManager->AddNtupleRow();
}

void EventAction::SetIncomingMomentum(G4ThreeVector p)
{
    fIncomingMomentum = p;
}


void EventAction::SetOutgoingMomentum(G4ThreeVector p)
{
    fOutgoingMomentum = p;
}

void EventAction::SetScatteringAngle(G4double angle)
{
    fScatteringAngle = angle;
}


G4double EventAction::GetScatteringAngle() const
{
    return fScatteringAngle;
}