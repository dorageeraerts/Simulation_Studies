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
  auto hce = event->GetHCofThisEvent();
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
  return hc;
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

void EventAction::EndOfEventAction(const G4Event* event)
{
    const char* clusterId = std::getenv("CLUSTER_ID");
const char* processId = std::getenv("PROCESS_ID");
G4int clusterIdInt = clusterId ? std::atoi(clusterId) : -1;
G4int processIdInt = processId ? std::atoi(processId) : -1;

auto analysisManager = G4AnalysisManager::Instance();
auto mytracking = TrackingAction::Instance();

G4int eventID = event->GetEventID();

auto hc = GetHC(event, fScintbarsHCID);
if ( !hc ) return;

auto nhit = hc->GetSize();

// Trigger condition: require a hit in BOTH panels (copy IDs 0 and 1)
bool panelHit[2] = {false, false};

for (unsigned long i = 0; i < nhit; i++) {
    auto hit = static_cast<ScintbarHit*>(hc->GetHit(i));
    int panelID = hit->GetPanelID();

    if (panelID == 0 || panelID == 1)
        panelHit[panelID] = true;
}

bool goodEvent = panelHit[0] && panelHit[1];

if (goodEvent) {
    // Loop over hits
    for ( unsigned long i = 0; i < nhit; i++ ) {
        auto hit = static_cast<ScintbarHit*>(hc->GetHit(i));
        auto entrypoint = hit->GetEntryPoint();
        auto exitpoint = hit->GetExitPoint();

        analysisManager->FillNtupleIColumn(2, 0, eventID);
        analysisManager->FillNtupleIColumn(2, 1, hit->GetParentId());
        analysisManager->FillNtupleDColumn(2, 2, hit->GetEdep());
        analysisManager->FillNtupleDColumn(2, 3, entrypoint.x());
        analysisManager->FillNtupleDColumn(2, 4, entrypoint.y());
        analysisManager->FillNtupleDColumn(2, 5, entrypoint.z());
        analysisManager->FillNtupleDColumn(2, 6, exitpoint.x());
        analysisManager->FillNtupleDColumn(2, 7, exitpoint.y());
        analysisManager->FillNtupleDColumn(2, 8, exitpoint.z());
        analysisManager->FillNtupleDColumn(2, 9, hit->GetPathLength());
        analysisManager->FillNtupleIColumn(2, 10, hit->GetPanelID());
        analysisManager->FillNtupleIColumn(2, 11, hit->GetPDGcode());
        analysisManager->FillNtupleIColumn(2, 12, hit->GetTrackID());
        analysisManager->FillNtupleIColumn(2, 13, clusterIdInt);
        analysisManager->FillNtupleIColumn(2, 14, processIdInt);
        analysisManager->FillNtupleDColumn(2, 15, hit->GetHitTime());

        analysisManager->AddNtupleRow(2);
    }

    // Loop over primaries — unchanged, still ntuple Id 1
    analysisManager->FillNtupleIColumn(1, 0, eventID);
    for ( int i = 0; i < event->GetNumberOfPrimaryVertex(); i++ ) {

        G4ThreeVector vertexPos = event->GetPrimaryVertex(i)->GetPosition();

        for ( int q = 0; q < event->GetPrimaryVertex(i)->GetNumberOfParticle(); q++ ) {

            auto primary = event->GetPrimaryVertex(i)->GetPrimary(q);

            analysisManager->FillNtupleIColumn(1, 1, primary->GetTrackID());
            analysisManager->FillNtupleIColumn(1, 2, primary->GetG4code()->GetPDGEncoding());
            analysisManager->FillNtupleDColumn(1, 3, primary->GetTotalEnergy());
            analysisManager->FillNtupleDColumn(1, 4, primary->GetMomentumDirection().theta());
            analysisManager->FillNtupleDColumn(1, 5, primary->GetMomentumDirection().phi());
            analysisManager->FillNtupleIColumn(1, 6, event->IsAborted());
            analysisManager->FillNtupleIColumn(1, 7, clusterIdInt);
            analysisManager->FillNtupleIColumn(1, 8, processIdInt);
            analysisManager->FillNtupleDColumn(1, 9, vertexPos.x() / mm);
            analysisManager->FillNtupleDColumn(1, 10, vertexPos.y() / mm);
            analysisManager->FillNtupleDColumn(1, 11, vertexPos.z() / mm);
        }
        analysisManager->AddNtupleRow(1);
    }
}
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