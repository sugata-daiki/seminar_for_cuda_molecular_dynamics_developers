#ifndef __FORCEFIELDS_H__
#define __FORCEFIELDS_H__

enum class BondedForceFieldType {
    None = 0,
    HarmonicBond;
};

template <bool EnableHarmonicBond = false>
struct EnabledBondedForceFields {};
#endif // __FORCEFIELDS_H__
