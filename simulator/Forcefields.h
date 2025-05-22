#ifndef __FORCEFIELDS_H__
#define __FORCEFIELDS_H__

enum class BondedForceField {
    HarmonicBond, 
    CosineAngle,
    Dihedral,
    Unknown
};

typedef struct {
    bool EnableHarmonicBond;
    bool EnableCosineAngle;
    bool EnableDihedral;
} EnableBondedForceFields;

template <bool EnableHarmonicBond = false, 
          bool EnableCosineAngle = false, 
          bool EnableDihedral = false>
struct EnableBondedForceFields_T {};
#endif // __FORCEFIELDS_H__
