/// \file DetectorConstruction.cc
/// \brief Implementation of the DetectorConstruction class

#include "DetectorConstruction.hh"

#include "G4Box.hh"
#include "G4Colour.hh"
#include "G4FieldManager.hh"
#include "G4GenericMessenger.hh"
#include "G4LogicalVolume.hh"
#include "G4Material.hh"
#include "G4NistManager.hh"
#include "G4PVParameterised.hh"
#include "G4PVPlacement.hh"
#include "G4PVReplica.hh"
#include "G4RotationMatrix.hh"
#include "G4RunManager.hh"
#include "G4SDManager.hh"
#include "G4NistManager.hh"
#include "G4SystemOfUnits.hh"
#include "G4ThreeVector.hh"
#include "G4Tubs.hh"
#include "G4UserLimits.hh"
#include "G4VisAttributes.hh"
#include "G4ExtrudedSolid.hh"
#include "G4GenericTrap.hh"
#include "G4IntersectionSolid.hh"
#include "G4SubtractionSolid.hh"
#include "G4VPrimitiveScorer.hh"
#include "G4MaterialPropertiesTable.hh"
#include "G4Element.hh"
#include "G4TessellatedSolid.hh"
 #include "G4Trap.hh"
#include <vector>

#include "ScintBarSD.hh"
#include "Materials.hh"

DetectorConstruction::DetectorConstruction(const char *detectorName)
  : G4VUserDetectorConstruction(),
    _nBars(32), _nModules(2), _nStations(4), _stationSpacing(50 * cm), _barLength(1070 * mm), _barHeight(17 * mm),
    _barBase(33. * mm), _triangEffectiveBase(32.5 * mm), _rotUpperX(NULL), _rotLowerX(NULL), _rotUpperY(NULL), _rotLowerY(NULL),
    _halfContLengthZ(0.), _halfContLengthXY(0.), _looseAccCheck(0.), _detType("triangular"), _cornerCut(1.*mm), _zPosStations({-0.26*m,0.*m,0.262*m,1.492*m}), _yPosStations({0.289*m,0.249*m,0.207*m,0*m})
{
  _messenger = new G4GenericMessenger(this, std::string("/muraves/").append(detectorName).append("/"));
  _messenger->DeclareProperty("nBars", _nBars, "Set the number of scintillating bars per module");
  _messenger->DeclareProperty("nModules", _nModules, "Set the number of modules per plane");
  _messenger->DeclareProperty("nStations", _nStations, "Set the number of XY stations.");
  _messenger->DeclarePropertyWithUnit("barLength","mm", _barLength, "Set the length of the scintillating bars.");
  _messenger->DeclarePropertyWithUnit("barHeight","mm", _barHeight, "Set the height of the scintillating bars.");
  _messenger->DeclarePropertyWithUnit("barBase","mm", _barBase, "Set the size of the base of the scintillating bars (this includes the gap).");
  _messenger->DeclarePropertyWithUnit("triangEffBase","mm", _triangEffectiveBase,
				      "Set the effective size of the base of the triangular bars due to gap (this base length does not include air gap).");
  _messenger->DeclarePropertyWithUnit("cornerCut","mm", _cornerCut,
				      "Set the size of the unusable (dead) tip of the triangular bar.");

      // because _zPos here becomes xPos IRL (rotation later on)
    _messenger->DeclarePropertyWithUnit("xPos0", "m", _zPosStations[0], "X position of station 0");
    _messenger->DeclarePropertyWithUnit("xPos1", "m", _zPosStations[1], "X position of station 1");
    _messenger->DeclarePropertyWithUnit("xPos2", "m", _zPosStations[2], "X position of station 2");
    _messenger->DeclarePropertyWithUnit("xPos3", "m", _zPosStations[3], "X position of station 3");

    // because _yPos here becomes zPos IRL (rotation later on)
    _messenger->DeclarePropertyWithUnit("zPos0", "m", _yPosStations[0], "Z position of station 0");
    _messenger->DeclarePropertyWithUnit("zPos1", "m", _yPosStations[1], "Z position of station 1");
    _messenger->DeclarePropertyWithUnit("zPos2", "m", _yPosStations[2], "Z position of station 2");
    _messenger->DeclarePropertyWithUnit("zPos3", "m", _yPosStations[3], "Z position of station 3");

  _messenger->DeclareProperty("looseAcceptanceCheck", _looseAccCheck,
			      "Detector-face-enlargement factor for acceptance check (e.g. 1.1 -> enlarge face size by 10% when checking acceptance, 1 -> no enlargement)");
  _messenger->DeclareProperty("detectorType", _detType,
			      "Set the detector type (triangular bars, square bars, monolithic layers)").SetCandidates("triangular square monolithic");

  Materials::makeMaterials();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

DetectorConstruction::~DetectorConstruction()
{
  delete _messenger;
  delete _rotUpperX;
  delete _rotLowerX;
  delete _rotUpperY;
  delete _rotLowerY;
  //if(fMaterials)  delete fMaterials; 
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4VPhysicalVolume* DetectorConstruction::Construct()
{
  //sfMaterials = Materials::GetInstance();

  xySafety = _barBase - _triangEffectiveBase; //  XY shift for gap inbetween bars
  G4cout << "[MuravesDetector::Construct] xySafety = " << xySafety << G4endl;
  zSafety = 5.5 * cm; //  distance between X and Y layer
  
  G4NistManager* nist = G4NistManager::Instance();
  G4Material* air_mat = nist->FindOrBuildMaterial("G4_AIR");
  G4Material* Aluminum_mat = nist->FindOrBuildMaterial("G4_Al");
  G4Material* polystyrene_mat = nist->FindOrBuildMaterial("G4_POLYSTYRENE"); // standard polystyrene
  G4Material* rock_mat = new G4Material("StandardRock", 11.0, 22.0*g/mole, 2.65*g/cm3);
  G4Material* plastic_mat = nist->FindOrBuildMaterial("G4_POLYVINYL_CHLORIDE"); // PVC plastic
  //G4Material* polystyrene_mat = FindMaterial(Materials::kPOLYSTYRENE); // with costum scintillator properties


  G4bool checkOverlaps = true; // Option to switch on/off checking of volume overlaps

       
    // ------------------- World -------------------
    
    G4double world_sizeX = 100.*m;
    G4double world_sizeY = 30.*m;
    G4double world_sizeZ  = 20*m;
    
    G4Box* solidWorld = new G4Box("World", 0.5*world_sizeX, 0.5*world_sizeY, 0.5*world_sizeZ);    

    G4LogicalVolume* logicWorld = new G4LogicalVolume(solidWorld, air_mat, "World");            

    G4VPhysicalVolume* physWorld = 
        new G4PVPlacement(0,           //no rotation
                G4ThreeVector(),       //at (0,0,0)
                logicWorld,            //its logical volume
                "World",               //its name
                0,                     //its mother  volume
                false,                 //no boolean operation
                0,                     //copy number
                checkOverlaps);        //overlaps checking


  // ------ Flank -------
 G4double width  = 100.*m;      // X direction
 G4double length = 30.*m;      // Y direction
G4double angle  = 10.*deg;

G4double height = width * std::tan(angle);  // Z height = 3.64 m


// Triangle in local XY plane:
// local x -> physical Y
// local y -> physical Z

std::vector<G4TwoVector> polygon;

polygon.emplace_back(-width/2, 0);       // bottom left
polygon.emplace_back( width/2, 0);       // bottom right
polygon.emplace_back( width/2, height);  // top


std::vector<G4ExtrudedSolid::ZSection> sections;

sections.emplace_back(
    -length/2,
    G4TwoVector(0,0),
    1.0
);

sections.emplace_back(
     length/2,
     G4TwoVector(0,0),
     1.0
);


G4ExtrudedSolid* solidFlank =
    new G4ExtrudedSolid(
        "Flank",
        polygon,
        sections
    );


// Rotate extrusion axis from local Z to global X
G4RotationMatrix* rot = new G4RotationMatrix();
rot->rotateX(-90.*deg);


G4LogicalVolume* logicFlank =
    new G4LogicalVolume(
        solidFlank,
        rock_mat,
        "flank"
    );


new G4PVPlacement(
    rot,
    G4ThreeVector(0,0,-0.5*world_sizeZ),
    logicFlank,
    "Flank",
    logicWorld,
    false,
    0,
    checkOverlaps
);

// ----- Scoring volume -----

// ------------------- Slope scoring layer -------------------

// Same dimensions as flank
G4double scorerThickness = 1.*mm;

// Outward shift to avoid overlap with rock
G4ThreeVector scorerOffset(
    -std::sin(angle) * scorerThickness,
     0,
     std::cos(angle) * scorerThickness
);


// Create thin triangular layer following the slope
std::vector<G4TwoVector> scorerPolygon;

scorerPolygon.emplace_back(
    -width/2.,
    0.
);

scorerPolygon.emplace_back(
     width/2.,
     height
);

scorerPolygon.emplace_back(
     width/2.,
     height + scorerThickness
);


// Extrusion along local Z (becomes physical Y)
std::vector<G4ExtrudedSolid::ZSection> scorerSections;

scorerSections.emplace_back(
    -length/2.,
    G4TwoVector(0,0),
    1.0
);

scorerSections.emplace_back(
     length/2.,
    G4TwoVector(0,0),
    1.0
);


G4ExtrudedSolid* solidSlopeScorer =
    new G4ExtrudedSolid(
        "SlopeScorer",
        scorerPolygon,
        scorerSections
    );



G4LogicalVolume* logicSlopeScorer =
    new G4LogicalVolume(
        solidSlopeScorer,
        air_mat,
        "SlopeScorer"
    );


// Same rotation as flank
new G4PVPlacement(
    rot,
    G4ThreeVector(0,0,-0.5*world_sizeZ) + scorerOffset,
    logicSlopeScorer,
    "SlopeScorer",
    logicWorld,
    false,
    0,
    checkOverlaps
);

  // ------------------- visualization attributes -------------------

    G4VisAttributes* whitecol = new G4VisAttributes(G4Colour(1.0,1.0,1.0));
    G4VisAttributes* pinkcol = new G4VisAttributes(G4Colour(0.6,0.0,0.6));
    G4VisAttributes* graycol = new G4VisAttributes(G4Colour(0.9,0.9,0.9));
    G4VisAttributes* cyancol = new G4VisAttributes(G4Colour(0.0,1.0,1.0,0.3));
    G4VisAttributes* redcol = new G4VisAttributes(G4Colour(0.5,0.0,0.0));
    G4VisAttributes* darkgraycol = new G4VisAttributes(G4Colour(0.8,0.8,0.8));
    G4VisAttributes* orangecol = new G4VisAttributes(G4Colour(0.8,0.5,0.));
    G4VisAttributes* yellowcol = new G4VisAttributes(G4Colour(1.0,1.0,0.));
    G4VisAttributes* greencol = new G4VisAttributes(G4Colour(0.,1.0,0.,0.4));
    G4VisAttributes* bluecol = new G4VisAttributes(G4Colour(0.,0.,0.8,0.5));

    //logicAlFoil->SetVisAttributes(bluecol);
    //logicAlShell->SetVisAttributes(cyancol);
    //logicTEC->SetVisAttributes(greencol);

  bluecol->SetForceSolid(true);
  //logicAlFoil->SetVisAttributes(bluecol);

  G4VisAttributes* scorerVis =
    new G4VisAttributes(
        G4Colour(0.,1.,0.,0.5)
    );

scorerVis->SetForceSolid(true);

logicSlopeScorer->SetVisAttributes(scorerVis);
     
  auto barVis = new G4VisAttributes(G4Colour(1.0, 0.0, 0.0, 0.4)); 
  barVis->SetForceSolid(true);  // fill volume color
  //barLog->SetVisAttributes(barVis);

  auto tapeVis = new G4VisAttributes(G4Colour(1.0, 1.0, 0.0, 0.5)); // yellow
  tapeVis->SetForceSolid(true);
  //tapeLog->SetVisAttributes(tapeVis);

  // ------------------- Return world -------------------
  return physWorld;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void DetectorConstruction::ConstructSDandField()
{
  /*auto sdManager = G4SDManager::GetSDMpointer();
  G4String SDname;
  auto Scintbars = new ScintbarSD(SDname="/Scintbars");
  sdManager->AddNewDetector(Scintbars);
  barLog->SetSensitiveDetector(Scintbars);*/
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

// Track must hit front and back layer
G4bool DetectorConstruction::IsInsideAcceptance(const G4ThreeVector &pos, const G4ThreeVector &dir) const
{
  // 1. Reject particles not entering from front side
  if (pos[2] < _halfContLengthZ)
    return false;

  // 2. Reject particles not hitting the front side
  float hitPoint = fabs(pos[0] + (dir[0] / dir[2]) * (_halfContLengthZ - pos[2])); // X coordinate
  if (hitPoint > _halfContLengthXY * _looseAccCheck)
    return false;
  hitPoint = fabs(pos[1] + (dir[1] / dir[2]) * (_halfContLengthZ - pos[2])); // Y coordinate
  if (hitPoint > _halfContLengthXY * _looseAccCheck)
    return false;

  // 3. Reject particles not hitting the back side
  hitPoint = fabs(pos[0] + (dir[0] / dir[2]) * (-_halfContLengthZ - pos[2])); // X coordinate
  if (hitPoint > _halfContLengthXY * _looseAccCheck)
    return false;
  hitPoint = fabs(pos[1] + (dir[1] / dir[2]) * (-_halfContLengthZ - pos[2])); // Y coordinate
  if (hitPoint > _halfContLengthXY * _looseAccCheck)
    return false;

  return true;
}

G4Material* DetectorConstruction::FindMaterial(G4String name) 
{
    G4Material* material = G4Material::GetMaterial(name,true);
    return material;
}

void DetectorConstruction::DeleteMessenger()
{
  delete _messenger;
  _messenger = NULL;
}