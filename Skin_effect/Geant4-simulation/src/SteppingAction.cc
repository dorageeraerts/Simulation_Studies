#include "SteppingAction.hh"
#include "EventAction.hh"

#include "G4Step.hh"
#include "G4Track.hh"
#include "G4RunManager.hh"
#include "G4SystemOfUnits.hh"
#include "G4AnalysisManager.hh"

#include <cmath>


MySteppingAction::MySteppingAction(EventAction* eventAction)
    : fEventAction(eventAction),
      fLastEventID(-1),
      fHardScatterThreshold(5.*deg)
{
}


MySteppingAction::~MySteppingAction()
{
}

void MySteppingAction::UserSteppingAction(const G4Step* step)
{
    G4Track* track = step->GetTrack();

    // -------------------------
    // Only muons
    // -------------------------
    const G4String& particleName = track->GetDefinition()->GetParticleName();
    if (particleName != "mu+" && particleName != "mu-")
    {
        return;
    }

    const G4int eventID =
        G4RunManager::GetRunManager()->GetCurrentEvent()->GetEventID();

    // -------------------------
    // KinkScorer diagnostic: log every step taken *inside* KinkScorer,
    // and flag hard deflections separately
    // -------------------------

    G4VPhysicalVolume* preVolume  = step->GetPreStepPoint()->GetPhysicalVolume();
    G4VPhysicalVolume* postVolume = step->GetPostStepPoint()->GetPhysicalVolume();

    // preVolume can be null on the very first step of a track in some edge
    // cases (e.g. primary vertex generation) - guard against that.
    G4bool inKinkScorer =
        preVolume && preVolume->GetName() == "KinkScorer";

    if (inKinkScorer)
    {
        G4ThreeVector dirIn  = step->GetPreStepPoint()->GetMomentumDirection();
        G4ThreeVector dirOut = step->GetPostStepPoint()->GetMomentumDirection();
        G4double dtheta = dirIn.angle(dirOut); // radians

        // Process that defined the end of this step (msc, CoulombScat,
        // Transportation for a boundary crossing, etc.)
        G4String procName = "Unknown";
        const G4VProcess* proc = step->GetPostStepPoint()->GetProcessDefinedStep();
        if (proc) procName = proc->GetProcessName();

        G4bool isHard = (dtheta > fHardScatterThreshold);

        G4ThreeVector pos = step->GetPostStepPoint()->GetPosition();
        G4double kineticEnergy = step->GetPreStepPoint()->GetKineticEnergy();

        auto* analysisManager = G4AnalysisManager::Instance();

        analysisManager->FillNtupleIColumn(0, 0, eventID);
        analysisManager->FillNtupleIColumn(0, 1, track->GetTrackID());
        analysisManager->FillNtupleDColumn(0, 2, dtheta/deg);
        analysisManager->FillNtupleSColumn(0, 3, procName);
        analysisManager->FillNtupleIColumn(0, 4, isHard ? 1 : 0);
        analysisManager->FillNtupleDColumn(0, 5, pos.x()/m);
        analysisManager->FillNtupleDColumn(0, 6, pos.y()/m);
        analysisManager->FillNtupleDColumn(0, 7, pos.z()/m);
        analysisManager->FillNtupleDColumn(0, 8, kineticEnergy/GeV);
        analysisManager->AddNtupleRow(0);
    }

    // -------------------------
    // Panel crossings, unchanged logic from before but now with a
    // real muon-type check backing it, kept independent of KinkScorer.
    // (Fill this in once you wire Panel1/Panel2 into an ntuple too -
    // for step 1 the KinkScorer block above is the one that matters.)
    // -------------------------

    if (!postVolume) return;

    if (postVolume->GetName() != "Panel1" && postVolume->GetName() != "Panel2")
    {
        return;
    }

    // placeholder for detector-hit recording, extend as needed
}