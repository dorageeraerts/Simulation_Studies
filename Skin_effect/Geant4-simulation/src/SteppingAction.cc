#include "SteppingAction.hh"
#include "EventAction.hh"

#include "G4Step.hh"
#include "G4Track.hh"
#include "G4RunManager.hh"
#include "G4SystemOfUnits.hh"

#include <cmath>


MySteppingAction::MySteppingAction(EventAction* eventAction)
    : fEventAction(eventAction),
      muonEntered(false),
      fLastEventID(-1),
      fScorerHits(0)
{
}


MySteppingAction::~MySteppingAction()
{
}

void MySteppingAction::UserSteppingAction(const G4Step* step)
{

    G4Track* track = step->GetTrack();


    // Only muons
    if(track->GetDefinition()->GetParticleName() != "mu+" &&
       track->GetDefinition()->GetParticleName() != "mu-")
    {
        return;
    }


    auto postVolume =
        step->GetPostStepPoint()
        ->GetTouchableHandle()
        ->GetVolume();


    if(!postVolume)
        return;



    // Check crossing of scorer
    if(postVolume->GetName() != "SlopeScorer")
    {
        return;
    }



    G4ThreeVector momentum =
        step->GetPreStepPoint()
        ->GetMomentum();



    fScorerHits++;


    // -------------------------
    // First crossing
    // -------------------------

    if(fScorerHits == 1)
    {

        fEventAction->SetIncomingMomentum(momentum);


        G4cout
        << "First scorer crossing"
        << G4endl;

        G4cout
        << "p_in = "
        << momentum
        << G4endl;

    }



    // -------------------------
    // Second crossing
    // -------------------------

    if(fScorerHits == 2)
    {

        fEventAction->SetOutgoingMomentum(momentum);


        G4double angle =
            fEventAction
            ->GetIncomingMomentum()
            .angle(momentum);


        fEventAction->SetScatteringAngle(angle);



        G4cout
        << "Second scorer crossing"
        << G4endl;


        G4cout
        << "p_out = "
        << momentum
        << G4endl;


        G4cout
        << "scattering angle = "
        << angle/deg
        << " deg"
        << G4endl;

    }

    const G4int eventID =
    G4RunManager::GetRunManager()
    ->GetCurrentEvent()
    ->GetEventID();

    if(eventID != fLastEventID)
{
    fLastEventID = eventID;

    // reset scorer crossings for new muon/event
    fScorerHits = 0;
}

}