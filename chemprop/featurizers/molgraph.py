from __future__ import annotations

from dataclasses import InitVar, dataclass, field, fields
from typing import List, Optional, Sequence, Tuple

import numpy as np
from chemprop.featurizers.atom import AtomFeaturizer
from chemprop.featurizers.bond import BondFeaturizer

from chemprop.rdkit import make_mol


@dataclass
class MolGraph:
    """A `MolGraph` represents the graph structure and featurization of a single molecule.

    Attributes
    ----------
    n_atoms : int
        the number of atoms in the molecule
    n_bonds : int
        the number of bonds in the molecule
    X_v : np.ndarray
        the atom features of the molecule
    X_e : np.ndarray
        the bond features of the molecule
    a2b : list[tuple[int]]
        A mapping from an atom index to a list of incoming bond indices.
    b2a : list[int]
        A mapping from a bond index to the index of the atom the bond originates from.
    b2revb
        A mapping from a bond index to the index of the reverse bond.
    """
    n_atoms: int
    n_bonds: int
    X_v: np.ndarray
    X_e: np.ndarray
    a2b: list[tuple[int]]
    b2a: list[int]
    b2revb: list[int]
    b2b: list[int]
    a2a: list[int]


@dataclass
class MolGraphFeaturizer:
    """A `MolGraphFeaturizer` featurizes molecules (in the form of SMILES strings) into `MolGraph`s

    NOTE: hydrogens are generally treated implicitly in `MolGraph`s unless `keep_h` or `add_h` are 
    set to `True`

    Attributes
    ----------
    atom_featurizer : AtomFeaturizer
    bond_featurizer : BondFeaturizer
    atom_fdim : int
        the dimension of atom feature represenatations in this featurizer
    bond_fdim : int
        the dimension of bond feature represenatations in this featurizer
    atom_messages : bool
    keep_h : bool
    add_h : bool

    Parameters
    ----------
    atom_featurizer : AtomFeaturizer, default=AtomFeaturizer(100)
        the featurizer with which to calculate feature representations of the atoms in a given 
        molecule
    bond_featurizer : BondFeaturizer, default=BondFeaturizer(14)
        the featurizer with which to calculate feature representations of the bonds in a given 
        molecule
    extra_atom_fdim : int, default=0
        the dimension of the additional features that will be concatenated onto the calculated 
        features of each atom
    extra_bond_fdim : int, default=0
        the dimension of the additional features that will be concatenated onto the calculated 
        features of each bond
    atom_messages : bool, default=False
        whether to prepare the `MolGraph` for use with atom-based messages
    keep_h : bool, default=False
        whether to retain the hydrogens present in input molecules or remove them from the prepared 
        structure
    add_h : bool, default=False
        whether to add hydrogens to all input molecules when preparing the input structure
    """
    atom_featurizer: AtomFeaturizer = field(default_factory=lambda: AtomFeaturizer(100))
    bond_featurizer: BondFeaturizer = field(default_factory=lambda: BondFeaturizer(14))
    atom_fdim: int = field(init=False)
    extra_atom_fdim: InitVar[int] = 0
    bond_fdim: int = field(init=False)
    extra_bond_fdim: InitVar[int] = 0
    atom_messages: bool = False
    keep_h: bool = False
    add_h: bool = False

    def __post_init__(self, extra_atom_fdim: int, extra_bond_fdim):
        self.atom_fdim = len(self.atom_featurizer) + extra_atom_fdim
        self.bond_fdim = len(self.bond_featurizer) + extra_bond_fdim
        if not self.atom_messages:
            self.bond_fdim += self.atom_fdim

    def featurize(
        self,
        smi: str,
        atom_features_extra: Optional[np.ndarray] = None,
        bond_features_extra: Optional[np.ndarray] = None,
    ) -> MolGraph:
        mol = make_mol(smi, self.keep_h, self.add_h)

        n_atoms = mol.GetNumAtoms() 
        n_bonds = mol.GetNumBonds()
        X_v = []
        X_e = []
        a2b = [[]] * n_atoms
        b2a = []
        b2revb = []

        if atom_features_extra is not None and len(atom_features_extra) != n_atoms:
            raise ValueError(
                "Input molecule must have same number of atoms as `len(atom_features_extra)`!"
                f"got: {n_atoms} and {len(atom_features_extra)}, respectively"
            )
        if bond_features_extra is not None and len(bond_features_extra) != n_bonds:
            raise ValueError(
                "Input molecule must have same number of bonds as `len(bond_features_extra)`!"
                f"got: {n_bonds} and {len(bond_features_extra)}, respectively"
            )
            
        X_v = np.stack([self.atom_featurizer(a) for a in mol.GetAtoms()])
        X_e = np.empty((2 * n_bonds, self.bond_fdim))

        if atom_features_extra is not None:
            X_v = np.hstack((X_v, atom_features_extra))

        i = 0
        for a1 in range(n_atoms):
            for a2 in range(a1 + 1, n_atoms):
                b = mol.GetBondBetweenAtoms(a1, a2)

                if b is None:
                    continue

                x_e = self.bond_featurizer(b)
                if bond_features_extra is not None:
                    x_e = np.concat((x_e, bond_features_extra[b.GetIdx()]))

                b12 = i
                b21 = b12 + 1

                if self.atom_messages:
                    X_e[b12] = x_e
                    X_e[b21] = x_e
                else:
                    X_e[b12] = np.concatenate((X_v[a1], x_e))
                    X_e[b21] = np.concatenate((X_v[a2], x_e))

                a2b[a2].append(b12)
                b2a.append(a1)
                a2b[a1].append(b21)
                b2a.append(a2)
                b2revb.append(b21)
                b2revb.append(b12)

                i += 2

        return MolGraph(n_atoms, 2 * n_bonds, X_v, X_e, a2b, b2a, b2revb, None, None)
