/// \file SteppingAction.hh
/// \brief Definition of the SteppingAction class

#ifndef SteppingAction_h
#define SteppingAction_h 1

#include "G4UserSteppingAction.hh"
#include "globals.hh"
#include "EventAction.hh"

class MySteppingAction : public G4UserSteppingAction {
public:
    MySteppingAction(EventAction* eventAction);
    ~MySteppingAction() override;
    void UserSteppingAction(const G4Step* step) override;
    
private:
    G4int fSteps = 0;
    G4int fLastEventID = -1; 
    EventAction* fEventAction = nullptr;

    G4double fHardScatterThreshold;
};

#endif