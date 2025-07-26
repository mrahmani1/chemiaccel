"""
Reaction Network Construction

Core classes for building and managing reaction networks from chemical reactions,
including nodes representing chemical species and edges representing reactions.
"""

import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors, rdchem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    Chem = None

logger = logging.getLogger(__name__)


class ReactionType(Enum):
    """Types of chemical reactions."""
    
    SUBSTITUTION = "substitution"
    ADDITION = "addition"
    ELIMINATION = "elimination"
    REARRANGEMENT = "rearrangement"
    OXIDATION = "oxidation"
    REDUCTION = "reduction"
    CYCLIZATION = "cyclization"
    RING_OPENING = "ring_opening"
    COUPLING = "coupling"
    CLEAVAGE = "cleavage"


class SpeciesType(Enum):
    """Types of chemical species in reaction networks."""
    
    REACTANT = "reactant"
    PRODUCT = "product"
    INTERMEDIATE = "intermediate"
    TRANSITION_STATE = "transition_state"
    CATALYST = "catalyst"
    SOLVENT = "solvent"


@dataclass
class ChemicalSpecies:
    """Represents a chemical species in the reaction network."""
    
    species_id: str
    smiles: str
    name: Optional[str] = None
    species_type: SpeciesType = SpeciesType.INTERMEDIATE
    molecular_weight: Optional[float] = None
    formal_charge: int = 0
    multiplicity: int = 1
    energy: Optional[float] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize computed properties."""
        if RDKIT_AVAILABLE and self.smiles:
            self._compute_rdkit_properties()
    
    def _compute_rdkit_properties(self) -> None:
        """Compute RDKit-based properties."""
        try:
            mol = Chem.MolFromSmiles(self.smiles)
            if mol is not None:
                if self.molecular_weight is None:
                    self.molecular_weight = rdMolDescriptors.CalcExactMolWt(mol)
                
                self.properties.update({
                    'num_atoms': mol.GetNumAtoms(),
                    'num_bonds': mol.GetNumBonds(),
                    'num_rings': rdMolDescriptors.CalcNumRings(mol),
                    'tpsa': rdMolDescriptors.CalcTPSA(mol),
                    'logp': rdMolDescriptors.CalcCrippenDescriptors(mol)[0]
                })
        except Exception as e:
            logger.warning(f"Error computing RDKit properties for {self.smiles}: {e}")
    
    def to_mol(self):
        """Convert to RDKit molecule object."""
        if RDKIT_AVAILABLE:
            return Chem.MolFromSmiles(self.smiles)
        return None
    
    def get_molecular_formula(self) -> str:
        """Get molecular formula."""
        if RDKIT_AVAILABLE:
            mol = self.to_mol()
            if mol is not None:
                return rdMolDescriptors.CalcMolFormula(mol)
        return ""


@dataclass
class ReactionNode:
    """
    Node in the reaction network representing a chemical species.
    """
    
    node_id: str
    species: ChemicalSpecies
    degree: int = 0  # Number of connections
    reactions_in: List[str] = field(default_factory=list)
    reactions_out: List[str] = field(default_factory=list)
    
    def add_incoming_reaction(self, reaction_id: str) -> None:
        """Add incoming reaction."""
        if reaction_id not in self.reactions_in:
            self.reactions_in.append(reaction_id)
            self.degree += 1
    
    def add_outgoing_reaction(self, reaction_id: str) -> None:
        """Add outgoing reaction."""
        if reaction_id not in self.reactions_out:
            self.reactions_out.append(reaction_id)
            self.degree += 1
    
    def remove_reaction(self, reaction_id: str) -> None:
        """Remove reaction from node."""
        if reaction_id in self.reactions_in:
            self.reactions_in.remove(reaction_id)
            self.degree -= 1
        if reaction_id in self.reactions_out:
            self.reactions_out.remove(reaction_id)
            self.degree -= 1


@dataclass
class ElementaryStep:
    """
    Represents an elementary reaction step with detailed mechanistic information.
    """
    
    step_id: str
    reactants: List[str]  # Species IDs
    products: List[str]   # Species IDs
    reaction_type: ReactionType
    activation_energy: Optional[float] = None
    rate_constant: Optional[float] = None
    temperature: float = 298.15
    pressure: float = 1.0
    catalyst: Optional[str] = None
    solvent: Optional[str] = None
    mechanism_description: str = ""
    bond_changes: List[Dict[str, Any]] = field(default_factory=list)
    atom_mapping: Dict[int, int] = field(default_factory=dict)
    
    def add_bond_formation(self, atom1: int, atom2: int, bond_order: float = 1.0) -> None:
        """Add bond formation to reaction."""
        self.bond_changes.append({
            'type': 'formation',
            'atoms': [atom1, atom2],
            'bond_order': bond_order
        })
    
    def add_bond_breaking(self, atom1: int, atom2: int, bond_order: float = 1.0) -> None:
        """Add bond breaking to reaction."""
        self.bond_changes.append({
            'type': 'breaking',
            'atoms': [atom1, atom2],
            'bond_order': bond_order
        })
    
    def get_reaction_smiles(self, species_dict: Dict[str, ChemicalSpecies]) -> str:
        """Get reaction SMILES representation."""
        reactant_smiles = [species_dict[r_id].smiles for r_id in self.reactants if r_id in species_dict]
        product_smiles = [species_dict[p_id].smiles for p_id in self.products if p_id in species_dict]
        
        reactants_str = ".".join(reactant_smiles)
        products_str = ".".join(product_smiles)
        
        return f"{reactants_str}>>{products_str}"
    
    def calculate_reaction_enthalpy(self, species_dict: Dict[str, ChemicalSpecies]) -> Optional[float]:
        """Calculate reaction enthalpy from species energies."""
        try:
            reactant_energies = [species_dict[r_id].energy for r_id in self.reactants 
                               if r_id in species_dict and species_dict[r_id].energy is not None]
            product_energies = [species_dict[p_id].energy for p_id in self.products 
                              if p_id in species_dict and species_dict[p_id].energy is not None]
            
            if reactant_energies and product_energies:
                return sum(product_energies) - sum(reactant_energies)
        except Exception as e:
            logger.warning(f"Error calculating reaction enthalpy: {e}")
        
        return None


@dataclass
class ReactionEdge:
    """
    Edge in the reaction network representing a reaction between species.
    """
    
    edge_id: str
    elementary_step: ElementaryStep
    source_nodes: List[str]  # Reactant node IDs
    target_nodes: List[str]  # Product node IDs
    weight: float = 1.0
    reversible: bool = True
    reverse_rate_constant: Optional[float] = None
    
    def get_forward_barrier(self) -> Optional[float]:
        """Get forward activation barrier."""
        return self.elementary_step.activation_energy
    
    def get_reverse_barrier(self) -> Optional[float]:
        """Get reverse activation barrier."""
        if self.reversible and self.elementary_step.activation_energy is not None:
            # Simplified: assume reverse barrier = forward barrier - reaction enthalpy
            # In practice, this would be calculated more precisely
            return self.elementary_step.activation_energy
        return None
    
    def get_equilibrium_constant(self) -> Optional[float]:
        """Calculate equilibrium constant."""
        if self.elementary_step.rate_constant and self.reverse_rate_constant:
            return self.elementary_step.rate_constant / self.reverse_rate_constant
        return None


class ReactionNetwork:
    """
    Main class representing a chemical reaction network as a directed graph.
    
    Nodes represent chemical species and edges represent elementary reactions.
    Supports analysis of reaction pathways, kinetics, and thermodynamics.
    """
    
    def __init__(self, network_id: str = None):
        self.network_id = network_id or str(uuid.uuid4())
        self.nodes: Dict[str, ReactionNode] = {}
        self.edges: Dict[str, ReactionEdge] = {}
        self.species: Dict[str, ChemicalSpecies] = {}
        
        # Network properties
        self.temperature = 298.15
        self.pressure = 1.0
        self.pH = 7.0
        self.ionic_strength = 0.0
        
        # Analysis results cache
        self._pathways_cache = {}
        self._centrality_cache = {}
        
    def add_species(self, species: ChemicalSpecies) -> str:
        """Add a chemical species to the network."""
        self.species[species.species_id] = species
        
        # Create corresponding node
        node = ReactionNode(
            node_id=species.species_id,
            species=species
        )
        self.nodes[species.species_id] = node
        
        logger.debug(f"Added species: {species.species_id} ({species.smiles})")
        return species.species_id
    
    def add_reaction(self, elementary_step: ElementaryStep) -> str:
        """Add an elementary reaction step to the network."""
        
        # Ensure all species exist
        all_species = elementary_step.reactants + elementary_step.products
        missing_species = [s_id for s_id in all_species if s_id not in self.species]
        
        if missing_species:
            raise ValueError(f"Missing species in network: {missing_species}")
        
        # Create reaction edge
        edge = ReactionEdge(
            edge_id=elementary_step.step_id,
            elementary_step=elementary_step,
            source_nodes=elementary_step.reactants,
            target_nodes=elementary_step.products
        )
        
        self.edges[elementary_step.step_id] = edge
        
        # Update node connections
        for reactant_id in elementary_step.reactants:
            if reactant_id in self.nodes:
                self.nodes[reactant_id].add_outgoing_reaction(elementary_step.step_id)
        
        for product_id in elementary_step.products:
            if product_id in self.nodes:
                self.nodes[product_id].add_incoming_reaction(elementary_step.step_id)
        
        # Clear caches
        self._pathways_cache.clear()
        self._centrality_cache.clear()
        
        logger.debug(f"Added reaction: {elementary_step.step_id}")
        return elementary_step.step_id
    
    def remove_species(self, species_id: str) -> bool:
        """Remove a species and associated reactions from the network."""
        if species_id not in self.species:
            return False
        
        # Remove all reactions involving this species
        reactions_to_remove = []
        
        if species_id in self.nodes:
            node = self.nodes[species_id]
            reactions_to_remove.extend(node.reactions_in)
            reactions_to_remove.extend(node.reactions_out)
        
        for reaction_id in set(reactions_to_remove):
            self.remove_reaction(reaction_id)
        
        # Remove species and node
        del self.species[species_id]
        if species_id in self.nodes:
            del self.nodes[species_id]
        
        logger.debug(f"Removed species: {species_id}")
        return True
    
    def remove_reaction(self, reaction_id: str) -> bool:
        """Remove a reaction from the network."""
        if reaction_id not in self.edges:
            return False
        
        edge = self.edges[reaction_id]
        
        # Update node connections
        for reactant_id in edge.source_nodes:
            if reactant_id in self.nodes:
                self.nodes[reactant_id].remove_reaction(reaction_id)
        
        for product_id in edge.target_nodes:
            if product_id in self.nodes:
                self.nodes[product_id].remove_reaction(reaction_id)
        
        # Remove edge
        del self.edges[reaction_id]
        
        # Clear caches
        self._pathways_cache.clear()
        self._centrality_cache.clear()
        
        logger.debug(f"Removed reaction: {reaction_id}")
        return True
    
    def find_pathways(self, start_species: str, end_species: str, 
                     max_length: int = 5) -> List[List[str]]:
        """
        Find reaction pathways between two species.
        
        Args:
            start_species: Starting species ID
            end_species: Target species ID
            max_length: Maximum pathway length
            
        Returns:
            List of pathways (sequences of reaction IDs)
        """
        cache_key = (start_species, end_species, max_length)
        if cache_key in self._pathways_cache:
            return self._pathways_cache[cache_key]
        
        pathways = []
        
        def dfs(current_species: str, path: List[str], visited_species: Set[str]) -> None:
            if len(path) > max_length:
                return
            
            if current_species == end_species and path:
                pathways.append(path.copy())
                return
            
            if current_species in self.nodes:
                node = self.nodes[current_species]
                
                for reaction_id in node.reactions_out:
                    if reaction_id not in path:  # Avoid reaction cycles
                        edge = self.edges[reaction_id]
                        
                        for product_id in edge.target_nodes:
                            if product_id not in visited_species:
                                path.append(reaction_id)
                                visited_species.add(product_id)
                                dfs(product_id, path, visited_species)
                                visited_species.remove(product_id)
                                path.pop()
        
        dfs(start_species, [], {start_species})
        
        self._pathways_cache[cache_key] = pathways
        return pathways
    
    def calculate_node_centrality(self, centrality_type: str = "degree") -> Dict[str, float]:
        """
        Calculate node centrality measures.
        
        Args:
            centrality_type: Type of centrality ("degree", "betweenness", "closeness")
            
        Returns:
            Dictionary mapping species ID to centrality value
        """
        if centrality_type in self._centrality_cache:
            return self._centrality_cache[centrality_type]
        
        centrality = {}
        
        if centrality_type == "degree":
            # Degree centrality
            max_degree = max(node.degree for node in self.nodes.values()) if self.nodes else 1
            for species_id, node in self.nodes.items():
                centrality[species_id] = node.degree / max_degree if max_degree > 0 else 0.0
        
        elif centrality_type == "betweenness":
            # Simplified betweenness centrality
            for species_id in self.nodes:
                betweenness = 0.0
                
                # Count paths passing through this node
                for start_id in self.nodes:
                    for end_id in self.nodes:
                        if start_id != end_id and start_id != species_id and end_id != species_id:
                            all_paths = self.find_pathways(start_id, end_id, max_length=4)
                            paths_through_node = 0
                            
                            for pathway in all_paths:
                                # Check if any reaction in pathway involves the species
                                involves_species = False
                                for reaction_id in pathway:
                                    edge = self.edges[reaction_id]
                                    if (species_id in edge.source_nodes or 
                                        species_id in edge.target_nodes):
                                        involves_species = True
                                        break
                                
                                if involves_species:
                                    paths_through_node += 1
                            
                            if all_paths:
                                betweenness += paths_through_node / len(all_paths)
                
                centrality[species_id] = betweenness
        
        else:
            # Default to degree centrality
            return self.calculate_node_centrality("degree")
        
        self._centrality_cache[centrality_type] = centrality
        return centrality
    
    def get_network_statistics(self) -> Dict[str, Any]:
        """Get comprehensive network statistics."""
        
        num_species = len(self.species)
        num_reactions = len(self.edges)
        
        # Calculate degree distribution
        degrees = [node.degree for node in self.nodes.values()]
        avg_degree = np.mean(degrees) if degrees else 0
        max_degree = max(degrees) if degrees else 0
        
        # Species type distribution
        species_types = {}
        for species in self.species.values():
            stype = species.species_type.value
            species_types[stype] = species_types.get(stype, 0) + 1
        
        # Reaction type distribution
        reaction_types = {}
        for edge in self.edges.values():
            rtype = edge.elementary_step.reaction_type.value
            reaction_types[rtype] = reaction_types.get(rtype, 0) + 1
        
        return {
            'num_species': num_species,
            'num_reactions': num_reactions,
            'avg_degree': avg_degree,
            'max_degree': max_degree,
            'degree_distribution': degrees,
            'species_type_distribution': species_types,
            'reaction_type_distribution': reaction_types,
            'network_density': (2 * num_reactions) / (num_species * (num_species - 1)) if num_species > 1 else 0
        }
    
    def export_to_graphml(self, filepath: str) -> None:
        """Export network to GraphML format."""
        try:
            with open(filepath, 'w') as f:
                f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
                f.write('<graphml xmlns="http://graphml.graphdrawing.org/xmlns"\n')
                f.write('         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n')
                f.write('         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns\n')
                f.write('         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">\n')
                
                # Define attributes
                f.write('  <key id="smiles" for="node" attr.name="smiles" attr.type="string"/>\n')
                f.write('  <key id="species_type" for="node" attr.name="species_type" attr.type="string"/>\n')
                f.write('  <key id="reaction_type" for="edge" attr.name="reaction_type" attr.type="string"/>\n')
                f.write('  <key id="activation_energy" for="edge" attr.name="activation_energy" attr.type="double"/>\n')
                
                f.write('  <graph id="reaction_network" edgedefault="directed">\n')
                
                # Write nodes
                for species_id, species in self.species.items():
                    f.write(f'    <node id="{species_id}">\n')
                    f.write(f'      <data key="smiles">{species.smiles}</data>\n')
                    f.write(f'      <data key="species_type">{species.species_type.value}</data>\n')
                    f.write(f'    </node>\n')
                
                # Write edges
                for edge_id, edge in self.edges.items():
                    for source in edge.source_nodes:
                        for target in edge.target_nodes:
                            f.write(f'    <edge id="{edge_id}_{source}_{target}" source="{source}" target="{target}">\n')
                            f.write(f'      <data key="reaction_type">{edge.elementary_step.reaction_type.value}</data>\n')
                            if edge.elementary_step.activation_energy is not None:
                                f.write(f'      <data key="activation_energy">{edge.elementary_step.activation_energy}</data>\n')
                            f.write(f'    </edge>\n')
                
                f.write('  </graph>\n')
                f.write('</graphml>\n')
            
            logger.info(f"Network exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting network: {e}")


class ReactionNetworkBuilder:
    """
    Builder class for constructing reaction networks from various input sources.
    """
    
    def __init__(self):
        self.network = ReactionNetwork()
        self.species_map = {}  # SMILES -> species_id mapping
        
    def add_reaction_smiles(self, reaction_smiles: str, reaction_type: ReactionType = None,
                          activation_energy: float = None) -> str:
        """
        Add reaction from reaction SMILES string.
        
        Args:
            reaction_smiles: Reaction SMILES (e.g., "CCO.CC(=O)O>>CC(=O)OCC.O")
            reaction_type: Type of reaction
            activation_energy: Activation energy in kJ/mol
            
        Returns:
            Reaction ID
        """
        try:
            # Parse reaction SMILES
            if ">>" not in reaction_smiles:
                raise ValueError("Invalid reaction SMILES: missing '>>'")
            
            reactant_smiles, product_smiles = reaction_smiles.split(">>")
            reactants = reactant_smiles.split(".")
            products = product_smiles.split(".")
            
            # Add species
            reactant_ids = []
            for smiles in reactants:
                species_id = self._add_species_from_smiles(smiles, SpeciesType.REACTANT)
                reactant_ids.append(species_id)
            
            product_ids = []
            for smiles in products:
                species_id = self._add_species_from_smiles(smiles, SpeciesType.PRODUCT)
                product_ids.append(species_id)
            
            # Create elementary step
            step_id = f"rxn_{str(uuid.uuid4())[:8]}"
            elementary_step = ElementaryStep(
                step_id=step_id,
                reactants=reactant_ids,
                products=product_ids,
                reaction_type=reaction_type or ReactionType.SUBSTITUTION,
                activation_energy=activation_energy
            )
            
            # Add to network
            return self.network.add_reaction(elementary_step)
            
        except Exception as e:
            logger.error(f"Error adding reaction from SMILES '{reaction_smiles}': {e}")
            raise
    
    def _add_species_from_smiles(self, smiles: str, species_type: SpeciesType) -> str:
        """Add species from SMILES string."""
        
        # Check if species already exists
        if smiles in self.species_map:
            existing_id = self.species_map[smiles]
            # Update species type if needed
            existing_species = self.network.species[existing_id]
            if existing_species.species_type == SpeciesType.INTERMEDIATE:
                existing_species.species_type = species_type
            return existing_id
        
        # Create new species
        species_id = f"species_{str(uuid.uuid4())[:8]}"
        species = ChemicalSpecies(
            species_id=species_id,
            smiles=smiles,
            species_type=species_type
        )
        
        self.network.add_species(species)
        self.species_map[smiles] = species_id
        
        return species_id
    
    def build_from_reaction_database(self, reactions: List[Dict[str, Any]]) -> ReactionNetwork:
        """
        Build network from a list of reaction dictionaries.
        
        Args:
            reactions: List of reaction dictionaries with keys:
                      - 'smiles': reaction SMILES
                      - 'type': reaction type (optional)
                      - 'activation_energy': activation energy (optional)
                      - 'rate_constant': rate constant (optional)
        """
        for reaction_data in reactions:
            reaction_smiles = reaction_data.get('smiles')
            if not reaction_smiles:
                continue
            
            reaction_type = reaction_data.get('type')
            if reaction_type and isinstance(reaction_type, str):
                try:
                    reaction_type = ReactionType(reaction_type.lower())
                except ValueError:
                    reaction_type = ReactionType.SUBSTITUTION
            
            activation_energy = reaction_data.get('activation_energy')
            
            try:
                self.add_reaction_smiles(
                    reaction_smiles=reaction_smiles,
                    reaction_type=reaction_type,
                    activation_energy=activation_energy
                )
            except Exception as e:
                logger.warning(f"Failed to add reaction {reaction_smiles}: {e}")
        
        logger.info(f"Built network with {len(self.network.species)} species and {len(self.network.edges)} reactions")
        return self.network
    
    def get_network(self) -> ReactionNetwork:
        """Get the constructed reaction network."""
        return self.network