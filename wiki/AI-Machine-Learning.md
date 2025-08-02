# AI & Machine Learning

ChemiAccel provides comprehensive AI-driven tools for molecular discovery, including generative models, reinforcement learning optimization, and advanced property prediction. This guide covers AI workflows for accelerated chemical discovery.

## Overview

The AI/ML module enables:
- **Generative Chemistry**: Novel molecule generation using GANs, VAEs, and transformer models
- **Reinforcement Learning**: Goal-directed molecular optimization
- **Active Learning**: Efficient exploration of chemical space
- **Multi-Objective Optimization**: Simultaneous optimization of multiple molecular properties
- **Uncertainty Quantification**: Confidence estimation for predictions
- **Inverse Design**: Design molecules with target properties

## Quick Start

### Basic Generative Chemistry

```python
from chemi_accel.ai import MolecularGAN, PropertyPredictor
from chemi_accel import Calculator

# Initialize AI models
generator = MolecularGAN(model_type="conditional")
predictor = PropertyPredictor()

# Train generator on molecular dataset
training_molecules = load_molecule_dataset()  # Your molecule dataset
training_descriptors = Calculator().pandas(training_molecules, descriptors="default")

# Train the generative model
generator.train(
    molecules=training_molecules,
    descriptors=training_descriptors,
    epochs=100,
    batch_size=64
)

# Generate new molecules with desired properties
target_properties = {'logp': 2.5, 'tpsa': 60, 'molecular_weight': 300}
generated_molecules = generator.generate_with_conditions(
    num_molecules=100,
    target_properties=target_properties,
    temperature=0.8
)

print(f"Generated {len(generated_molecules)} novel molecules")
```

## Generative Models

### Molecular GANs

```python
from chemi_accel.ai import MolecularGAN, GANTrainer
import pandas as pd

class ConditionalMolecularGAN:
    """Conditional GAN for property-guided molecule generation."""
    
    def __init__(self, latent_dim=100, condition_dim=10):
        self.gan = MolecularGAN(
            architecture="conditional",
            latent_dim=latent_dim,
            condition_dim=condition_dim,
            vocab_size=50  # SMILES vocabulary size
        )
        self.trainer = GANTrainer()
    
    def train(self, molecules, properties, epochs=200):
        """Train conditional GAN on molecules and properties."""
        
        # Prepare training data
        training_data = self.trainer.prepare_training_data(
            molecules=molecules,
            properties=properties,
            sequence_length=128
        )
        
        # Training configuration
        config = {
            'learning_rate_g': 0.0002,
            'learning_rate_d': 0.0002,
            'beta1': 0.5,
            'batch_size': 64,
            'gradient_penalty': 10.0,
            'property_loss_weight': 1.0
        }
        
        # Train the model
        training_history = self.trainer.train(
            gan_model=self.gan,
            training_data=training_data,
            epochs=epochs,
            config=config,
            validation_split=0.1
        )
        
        return training_history
    
    def generate_molecules(self, target_properties, num_samples=100):
        """Generate molecules with target properties."""
        
        generated = self.gan.conditional_generate(
            conditions=target_properties,
            num_samples=num_samples,
            temperature=0.8,
            top_k=10
        )
        
        # Filter valid molecules
        valid_molecules = []
        for smiles in generated:
            if self.gan.validate_molecule(smiles):
                valid_molecules.append(smiles)
        
        return valid_molecules

# Example: Drug-like molecule generation
drug_molecules = load_drugbank_molecules()
drug_properties = Calculator().pandas(drug_molecules, descriptors=["molecular_weight", "logp", "tpsa", "hbd", "hba"])

# Train conditional GAN
cgan = ConditionalMolecularGAN()
training_history = cgan.train(drug_molecules, drug_properties, epochs=150)

# Generate drug-like molecules
target_drug_props = {
    'molecular_weight': 350,
    'logp': 2.0,
    'tpsa': 80,
    'hbd': 2,
    'hba': 4
}

new_drug_candidates = cgan.generate_molecules(target_drug_props, num_samples=500)
print(f"Generated {len(new_drug_candidates)} drug-like candidates")
```

### Variational Autoencoders (VAEs)

```python
from chemi_accel.ai import MolecularVAE

class PropertyOptimizedVAE:
    """VAE for molecular generation with property optimization."""
    
    def __init__(self, latent_dim=128, property_dim=5):
        self.vae = MolecularVAE(
            encoder_dim=256,
            decoder_dim=256,
            latent_dim=latent_dim,
            property_dim=property_dim
        )
    
    def train(self, molecules, properties, epochs=100):
        """Train VAE with property prediction."""
        
        training_data = self.vae.prepare_data(molecules, properties)
        
        # Training with property loss
        history = self.vae.train(
            training_data,
            epochs=epochs,
            batch_size=128,
            kl_weight=0.1,
            property_weight=1.0,
            reconstruction_weight=1.0
        )
        
        return history
    
    def optimize_molecules(self, starting_molecules, target_properties, 
                          optimization_steps=100):
        """Optimize molecules in latent space."""
        
        optimized_molecules = []
        
        for start_mol in starting_molecules:
            # Encode to latent space
            latent_vector = self.vae.encode(start_mol)
            
            # Optimize in latent space using gradient descent
            optimized_latent = self.vae.optimize_latent(
                initial_latent=latent_vector,
                target_properties=target_properties,
                steps=optimization_steps,
                learning_rate=0.01
            )
            
            # Decode back to molecule
            optimized_mol = self.vae.decode(optimized_latent)
            
            if self.vae.validate_molecule(optimized_mol):
                optimized_molecules.append(optimized_mol)
        
        return optimized_molecules

# Example: Optimize molecules for drug-likeness
vae = PropertyOptimizedVAE()
vae.train(drug_molecules, drug_properties, epochs=100)

# Start with lead compounds and optimize
lead_compounds = ["CCc1ccc(C(=O)O)cc1", "CC(C)Cc1ccc(C(C)C(=O)O)cc1"]
target_props = {'molecular_weight': 300, 'logp': 2.5, 'tpsa': 60}

optimized_leads = vae.optimize_molecules(lead_compounds, target_props)
print(f"Optimized {len(optimized_leads)} lead compounds")
```

### Transformer-Based Generation

```python
from chemi_accel.ai import MolecularTransformer

class ConditionalMolecularTransformer:
    """Transformer model for conditional molecule generation."""
    
    def __init__(self, vocab_size=50, embed_dim=256, num_heads=8, num_layers=6):
        self.transformer = MolecularTransformer(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            max_length=128,
            condition_dim=10
        )
    
    def train(self, molecules, conditions, epochs=50):
        """Train transformer on molecules with conditions."""
        
        # Tokenize molecules
        tokenized_data = self.transformer.tokenize_molecules(molecules)
        
        # Prepare conditioning data
        condition_vectors = self.transformer.prepare_conditions(conditions)
        
        # Training loop
        history = self.transformer.train(
            sequences=tokenized_data,
            conditions=condition_vectors,
            epochs=epochs,
            batch_size=32,
            learning_rate=1e-4,
            warmup_steps=4000
        )
        
        return history
    
    def generate_with_scaffold(self, scaffold, target_properties, num_samples=50):
        """Generate molecules with a given scaffold and target properties."""
        
        # Encode scaffold and properties
        scaffold_tokens = self.transformer.tokenize_molecule(scaffold)
        condition_vector = self.transformer.encode_properties(target_properties)
        
        # Generate completions
        generated_molecules = self.transformer.generate_with_prompt(
            prompt_tokens=scaffold_tokens,
            condition=condition_vector,
            num_samples=num_samples,
            max_new_tokens=64,
            temperature=0.8,
            top_p=0.9
        )
        
        return generated_molecules

# Example: Scaffold-based generation
transformer = ConditionalMolecularTransformer()
transformer.train(drug_molecules, drug_properties, epochs=30)

# Generate molecules with benzene scaffold
benzene_scaffold = "c1ccccc1"
target_props = {'molecular_weight': 400, 'logp': 3.0}

scaffold_derivatives = transformer.generate_with_scaffold(
    scaffold=benzene_scaffold,
    target_properties=target_props,
    num_samples=100
)

print(f"Generated {len(scaffold_derivatives)} benzene derivatives")
```

## Reinforcement Learning

### Molecular Optimization with RL

```python
from chemi_accel.ai import ReinforcementLearningOptimizer, MolecularEnvironment

class MolecularOptimizationRL:
    """Reinforcement learning for molecular optimization."""
    
    def __init__(self, action_space_size=50, state_dim=100):
        self.env = MolecularEnvironment(
            action_space_size=action_space_size,
            max_molecule_length=100,
            reward_function="multi_objective"
        )
        
        self.rl_optimizer = ReinforcementLearningOptimizer(
            algorithm="PPO",
            policy_network="transformer",
            state_dim=state_dim,
            action_dim=action_space_size
        )
    
    def define_reward_function(self, objectives):
        """Define multi-objective reward function."""
        
        def reward_function(molecule):
            if not self.env.validate_molecule(molecule):
                return -10.0  # Invalid molecule penalty
            
            # Calculate molecular properties
            calc = Calculator()
            props = calc.pandas([molecule], descriptors=list(objectives.keys())).iloc[0]
            
            # Calculate reward based on objectives
            total_reward = 0
            for prop_name, (target, weight, tolerance) in objectives.items():
                prop_value = props[prop_name]
                
                # Gaussian reward function
                deviation = abs(prop_value - target) / tolerance
                reward = weight * np.exp(-0.5 * deviation**2)
                total_reward += reward
            
            # Bonus for drug-likeness
            if self.env.passes_lipinski_rule_of_five(molecule):
                total_reward += 2.0
            
            # Penalty for toxic functional groups
            if self.env.contains_toxic_groups(molecule):
                total_reward -= 5.0
            
            return total_reward
        
        self.env.set_reward_function(reward_function)
        return reward_function
    
    def optimize(self, starting_molecules, objectives, episodes=1000):
        """Optimize molecules using reinforcement learning."""
        
        # Define reward function
        reward_func = self.define_reward_function(objectives)
        
        # Initialize RL agent
        agent = self.rl_optimizer.create_agent(
            environment=self.env,
            policy_lr=1e-4,
            value_lr=1e-3,
            entropy_coefficient=0.01
        )
        
        optimization_results = []
        
        for episode in range(episodes):
            # Select random starting molecule
            start_mol = np.random.choice(starting_molecules)
            
            # Run episode
            episode_result = agent.run_episode(
                initial_state=start_mol,
                max_steps=50
            )
            
            optimization_results.append({
                'episode': episode,
                'starting_molecule': start_mol,
                'final_molecule': episode_result['final_molecule'],
                'total_reward': episode_result['total_reward'],
                'steps': episode_result['steps']
            })
            
            # Log progress
            if episode % 100 == 0:
                avg_reward = np.mean([r['total_reward'] for r in optimization_results[-100:]])
                print(f"Episode {episode}, Average Reward: {avg_reward:.3f}")
        
        return optimization_results

# Example: Optimize for drug-like properties
rl_optimizer = MolecularOptimizationRL()

# Define multi-objective optimization
drug_objectives = {
    'molecular_weight': (350, 2.0, 50),    # target, weight, tolerance
    'logp': (2.5, 1.5, 1.0),
    'tpsa': (80, 1.0, 20),
    'hbd': (2, 0.5, 1),
    'hba': (4, 0.5, 2),
    'rotatable_bonds': (5, 0.5, 2)
}

# Starting molecules (lead compounds)
lead_compounds = [
    "CCc1ccc(C(=O)O)cc1",      # Simple carboxylic acid
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",  # Ibuprofen-like
    "c1ccc(N)cc1"              # Aniline
]

# Run optimization
optimization_results = rl_optimizer.optimize(
    starting_molecules=lead_compounds,
    objectives=drug_objectives,
    episodes=2000
)

# Analyze results
results_df = pd.DataFrame(optimization_results)
best_molecules = results_df.nlargest(10, 'total_reward')
print("Top 10 Optimized Molecules:")
print(best_molecules[['final_molecule', 'total_reward']])
```

### Multi-Agent Reinforcement Learning

```python
from chemi_accel.ai import MultiAgentRL, CollaborativeOptimization

class CollaborativeMolecularDesign:
    """Multi-agent system for collaborative molecular design."""
    
    def __init__(self, num_agents=4, specializations=None):
        self.num_agents = num_agents
        self.specializations = specializations or [
            "drug_likeness", "synthetic_accessibility", 
            "novelty", "target_affinity"
        ]
        
        self.multi_agent_system = MultiAgentRL(
            num_agents=num_agents,
            communication_dim=64,
            shared_memory_size=1000
        )
    
    def create_specialized_agents(self):
        """Create agents specialized for different objectives."""
        
        agents = []
        for i, specialization in enumerate(self.specializations):
            
            if specialization == "drug_likeness":
                reward_weights = {
                    'lipinski_violations': -2.0,
                    'lead_likeness': 1.5,
                    'oral_bioavailability': 1.0
                }
            elif specialization == "synthetic_accessibility":
                reward_weights = {
                    'sa_score': -1.0,  # Lower SA score is better
                    'retrosynthetic_complexity': -0.5,
                    'commercial_availability': 1.0
                }
            elif specialization == "novelty":
                reward_weights = {
                    'tanimoto_similarity': -1.0,  # Reward dissimilarity
                    'scaffold_novelty': 2.0,
                    'functional_group_diversity': 1.0
                }
            elif specialization == "target_affinity":
                reward_weights = {
                    'predicted_binding_affinity': 2.0,
                    'selectivity_score': 1.5,
                    'admet_properties': 1.0
                }
            
            agent = self.multi_agent_system.create_agent(
                agent_id=i,
                specialization=specialization,
                reward_weights=reward_weights
            )
            agents.append(agent)
        
        return agents
    
    def collaborative_optimization(self, target_protein, starting_molecules, 
                                 episodes=1000, collaboration_freq=10):
        """Run collaborative optimization with multiple agents."""
        
        agents = self.create_specialized_agents()
        shared_memory = self.multi_agent_system.initialize_shared_memory()
        
        optimization_results = []
        
        for episode in range(episodes):
            episode_molecules = []
            
            # Each agent proposes modifications
            for agent in agents:
                start_mol = np.random.choice(starting_molecules)
                
                # Agent generates molecule based on its specialization
                proposed_mol = agent.generate_molecule(
                    starting_molecule=start_mol,
                    shared_memory=shared_memory,
                    target_protein=target_protein
                )
                
                episode_molecules.append({
                    'agent_id': agent.agent_id,
                    'specialization': agent.specialization,
                    'molecule': proposed_mol,
                    'confidence': agent.get_confidence_score(proposed_mol)
                })
            
            # Collaborative evaluation and selection
            if episode % collaboration_freq == 0:
                # Agents share knowledge and update strategies
                self.multi_agent_system.collaborative_update(
                    agents=agents,
                    shared_memory=shared_memory,
                    recent_results=episode_molecules
                )
            
            # Evaluate all proposed molecules
            for mol_data in episode_molecules:
                evaluation = self.evaluate_molecule_comprehensive(
                    molecule=mol_data['molecule'],
                    target_protein=target_protein,
                    agent_specialization=mol_data['specialization']
                )
                
                mol_data.update(evaluation)
            
            optimization_results.extend(episode_molecules)
            
            # Log progress
            if episode % 100 == 0:
                best_scores = [max([m['total_score'] for m in optimization_results[-100:]]) for _ in range(4)]
                print(f"Episode {episode}, Best Scores by Agent: {best_scores}")
        
        return optimization_results
    
    def evaluate_molecule_comprehensive(self, molecule, target_protein, agent_specialization):
        """Comprehensive molecule evaluation."""
        
        calc = Calculator()
        props = calc.pandas([molecule], descriptors="comprehensive").iloc[0]
        
        # Basic property evaluation
        evaluation = {
            'molecular_weight': props['molecular_weight'],
            'logp': props['logp'],
            'tpsa': props['tpsa'],
            'sa_score': self.calculate_sa_score(molecule),
            'novelty_score': self.calculate_novelty_score(molecule),
        }
        
        # Target-specific evaluation
        if target_protein:
            binding_score = self.predict_binding_affinity(molecule, target_protein)
            evaluation['binding_affinity'] = binding_score
        
        # Calculate total score based on agent specialization
        if agent_specialization == "drug_likeness":
            total_score = self.calculate_drug_likeness_score(evaluation)
        elif agent_specialization == "synthetic_accessibility":
            total_score = self.calculate_synthesis_score(evaluation)
        elif agent_specialization == "novelty":
            total_score = self.calculate_novelty_total_score(evaluation)
        elif agent_specialization == "target_affinity":
            total_score = self.calculate_affinity_score(evaluation)
        
        evaluation['total_score'] = total_score
        return evaluation

# Example: Collaborative drug design for COVID-19 main protease
collaborative_design = CollaborativeMolecularDesign(
    num_agents=4,
    specializations=["drug_likeness", "synthetic_accessibility", "novelty", "target_affinity"]
)

# Target protein and starting molecules
covid_main_protease = load_protein_structure("6LU7")  # COVID-19 Mpro
starting_molecules = [
    "CC(C)CC(NC(=O)C1CCCN1C(=O)C(CC2=CC=CC=C2)NC(=O)C3=NC=CN=C3)C(=O)NC(CC4=CC=C(O)C=C4)C(=O)O",  # Known inhibitor
    "CC1=NN(C(=O)C1)C2=CC=CC=C2",  # Fragment
    "C1=CC=C(C=C1)C(=O)O"  # Benzoic acid
]

# Run collaborative optimization
collaborative_results = collaborative_design.collaborative_optimization(
    target_protein=covid_main_protease,
    starting_molecules=starting_molecules,
    episodes=1500,
    collaboration_freq=20
)

# Analyze results by agent specialization
results_df = pd.DataFrame(collaborative_results)
for spec in collaborative_design.specializations:
    agent_results = results_df[results_df['specialization'] == spec]
    best_mol = agent_results.loc[agent_results['total_score'].idxmax()]
    print(f"\nBest {spec} molecule: {best_mol['molecule']}")
    print(f"Score: {best_mol['total_score']:.3f}")
```

## Active Learning

### Uncertainty-Guided Exploration

```python
from chemi_accel.ai import ActiveLearner, UncertaintyQuantifier

class UncertaintyGuidedDiscovery:
    """Active learning system for molecular discovery."""
    
    def __init__(self, initial_labeled_size=1000, acquisition_batch_size=100):
        self.active_learner = ActiveLearner(
            base_estimator="neural_network",
            uncertainty_method="monte_carlo_dropout",
            acquisition_function="expected_improvement"
        )
        
        self.uncertainty_quantifier = UncertaintyQuantifier(
            method="ensemble",
            n_estimators=10
        )
        
        self.initial_labeled_size = initial_labeled_size
        self.acquisition_batch_size = acquisition_batch_size
    
    def initialize_active_learning(self, molecule_pool, property_name):
        """Initialize active learning with small labeled dataset."""
        
        # Calculate descriptors for molecule pool
        calc = Calculator()
        X_pool = calc.pandas(molecule_pool, descriptors="comprehensive")
        
        # Randomly select initial labeled set
        initial_indices = np.random.choice(
            len(molecule_pool), 
            self.initial_labeled_size, 
            replace=False
        )
        
        # Split into labeled and unlabeled sets
        X_labeled = X_pool.iloc[initial_indices]
        X_unlabeled = X_pool.drop(initial_indices)
        
        molecules_labeled = [molecule_pool[i] for i in initial_indices]
        molecules_unlabeled = [molecule_pool[i] for i in range(len(molecule_pool)) if i not in initial_indices]
        
        # Get experimental data for labeled set (simulate with noise)
        y_labeled = self.simulate_experimental_data(molecules_labeled, property_name)
        
        return X_labeled, y_labeled, X_unlabeled, molecules_labeled, molecules_unlabeled
    
    def active_learning_loop(self, molecule_pool, property_name, iterations=20):
        """Run active learning optimization loop."""
        
        # Initialize
        X_labeled, y_labeled, X_unlabeled, mols_labeled, mols_unlabeled = \
            self.initialize_active_learning(molecule_pool, property_name)
        
        learning_history = []
        
        for iteration in range(iterations):
            print(f"Active Learning Iteration {iteration + 1}")
            
            # Train ensemble model on current labeled data
            self.uncertainty_quantifier.fit(X_labeled, y_labeled)
            
            # Predict with uncertainty for unlabeled data
            predictions = self.uncertainty_quantifier.predict_with_uncertainty(X_unlabeled)
            
            # Select most informative samples
            acquisition_scores = self.calculate_acquisition_scores(
                predictions=predictions['mean'],
                uncertainties=predictions['std'],
                acquisition_function="upper_confidence_bound"
            )
            
            # Select top candidates
            selected_indices = np.argsort(acquisition_scores)[-self.acquisition_batch_size:]
            
            # "Perform experiments" (simulate)
            selected_molecules = [mols_unlabeled[i] for i in selected_indices]
            new_y_values = self.simulate_experimental_data(selected_molecules, property_name)
            
            # Update labeled and unlabeled sets
            new_X_labeled = X_unlabeled.iloc[selected_indices]
            X_labeled = pd.concat([X_labeled, new_X_labeled], ignore_index=True)
            y_labeled = np.concatenate([y_labeled, new_y_values])
            
            mols_labeled.extend(selected_molecules)
            
            # Remove selected samples from unlabeled set
            X_unlabeled = X_unlabeled.drop(X_unlabeled.index[selected_indices])
            mols_unlabeled = [mol for i, mol in enumerate(mols_unlabeled) if i not in selected_indices]
            
            # Record learning progress
            model_performance = self.evaluate_model_performance(X_labeled, y_labeled)
            
            learning_history.append({
                'iteration': iteration + 1,
                'labeled_size': len(X_labeled),
                'unlabeled_size': len(X_unlabeled),
                'model_r2': model_performance['r2_score'],
                'model_mae': model_performance['mae'],
                'best_predicted_value': predictions['mean'].max(),
                'max_uncertainty': predictions['std'].max(),
                'selected_molecules': selected_molecules
            })
            
            print(f"  Labeled samples: {len(X_labeled)}")
            print(f"  Model R²: {model_performance['r2_score']:.3f}")
            print(f"  Best predicted {property_name}: {predictions['mean'].max():.3f}")
        
        return learning_history, X_labeled, y_labeled, mols_labeled
    
    def calculate_acquisition_scores(self, predictions, uncertainties, acquisition_function):
        """Calculate acquisition function scores."""
        
        if acquisition_function == "uncertainty_sampling":
            return uncertainties
        
        elif acquisition_function == "upper_confidence_bound":
            beta = 2.0  # Exploration parameter
            return predictions + beta * uncertainties
        
        elif acquisition_function == "expected_improvement":
            current_best = predictions.max()
            z = (predictions - current_best) / (uncertainties + 1e-9)
            ei = (predictions - current_best) * norm.cdf(z) + uncertainties * norm.pdf(z)
            return ei
        
        elif acquisition_function == "probability_of_improvement":
            current_best = predictions.max()
            z = (predictions - current_best) / (uncertainties + 1e-9)
            return norm.cdf(z)
    
    def simulate_experimental_data(self, molecules, property_name, noise_std=0.1):
        """Simulate experimental measurements with noise."""
        
        # Calculate "true" values using high-fidelity calculation
        calc = Calculator()
        true_values = calc.pandas(molecules, descriptors=[property_name])[property_name].values
        
        # Add experimental noise
        noise = np.random.normal(0, noise_std, len(true_values))
        experimental_values = true_values + noise
        
        return experimental_values

# Example: Active learning for solubility prediction
active_discovery = UncertaintyGuidedDiscovery(
    initial_labeled_size=500,
    acquisition_batch_size=50
)

# Large unlabeled molecule pool
large_molecule_pool = load_large_molecule_database()  # 100K+ molecules

# Run active learning for solubility
solubility_history, final_X, final_y, final_molecules = active_discovery.active_learning_loop(
    molecule_pool=large_molecule_pool,
    property_name="solubility",
    iterations=25
)

# Analyze learning efficiency
history_df = pd.DataFrame(solubility_history)

# Plot learning curve
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(history_df['iteration'], history_df['model_r2'])
plt.title('Model Performance')
plt.ylabel('R² Score')
plt.xlabel('Iteration')

plt.subplot(2, 2, 2)
plt.plot(history_df['iteration'], history_df['labeled_size'])
plt.title('Dataset Growth')
plt.ylabel('Labeled Samples')
plt.xlabel('Iteration')

plt.subplot(2, 2, 3)
plt.plot(history_df['iteration'], history_df['best_predicted_value'])
plt.title('Best Discovered Property Value')
plt.ylabel('Solubility')
plt.xlabel('Iteration')

plt.subplot(2, 2, 4)
plt.plot(history_df['iteration'], history_df['max_uncertainty'])
plt.title('Maximum Uncertainty')
plt.ylabel('Prediction Uncertainty')
plt.xlabel('Iteration')

plt.tight_layout()
plt.show()

print(f"Final model trained on {len(final_X)} molecules")
print(f"Final R² score: {history_df['model_r2'].iloc[-1]:.3f}")
```

## Multi-Objective Optimization

### Pareto Optimization

```python
from chemi_accel.ai import MultiObjectiveOptimizer, ParetoFrontAnalysis

class MolecularMultiObjectiveOptimization:
    """Multi-objective optimization for molecular design."""
    
    def __init__(self, objectives, population_size=100):
        self.objectives = objectives
        self.population_size = population_size
        
        self.optimizer = MultiObjectiveOptimizer(
            algorithm="NSGA-II",
            population_size=population_size,
            crossover_prob=0.8,
            mutation_prob=0.1
        )
        
        self.pareto_analyzer = ParetoFrontAnalysis()
    
    def evaluate_population(self, molecules):
        """Evaluate molecular population for all objectives."""
        
        calc = Calculator()
        
        # Calculate all required descriptors
        descriptor_names = []
        for obj in self.objectives:
            descriptor_names.extend(obj['descriptors'])
        
        descriptor_names = list(set(descriptor_names))  # Remove duplicates
        descriptors_df = calc.pandas(molecules, descriptors=descriptor_names)
        
        # Evaluate each objective
        objective_values = []
        
        for i, molecule in enumerate(molecules):
            mol_descriptors = descriptors_df.iloc[i]
            obj_values = []
            
            for objective in self.objectives:
                obj_value = self.evaluate_single_objective(mol_descriptors, objective)
                obj_values.append(obj_value)
            
            objective_values.append(obj_values)
        
        return np.array(objective_values)
    
    def evaluate_single_objective(self, descriptors, objective):
        """Evaluate single objective function."""
        
        obj_type = objective['type']
        obj_params = objective['parameters']
        
        if obj_type == 'target_value':
            # Minimize distance to target value
            descriptor_name = obj_params['descriptor']
            target_value = obj_params['target']
            weight = obj_params.get('weight', 1.0)
            
            actual_value = descriptors[descriptor_name]
            distance = abs(actual_value - target_value)
            
            return -weight * distance  # Negative because we maximize
        
        elif obj_type == 'range_satisfaction':
            # Maximize satisfaction of range constraints
            total_satisfaction = 0
            
            for constraint in obj_params['constraints']:
                descriptor_name = constraint['descriptor']
                min_val = constraint['min']
                max_val = constraint['max']
                weight = constraint.get('weight', 1.0)
                
                actual_value = descriptors[descriptor_name]
                
                if min_val <= actual_value <= max_val:
                    satisfaction = weight
                else:
                    # Penalty based on distance from range
                    if actual_value < min_val:
                        penalty = (min_val - actual_value) / min_val
                    else:
                        penalty = (actual_value - max_val) / max_val
                    
                    satisfaction = weight * max(0, 1 - penalty)
                
                total_satisfaction += satisfaction
            
            return total_satisfaction
        
        elif obj_type == 'drug_likeness':
            # Drug-likeness score
            mw = descriptors['molecular_weight']
            logp = descriptors['logp']
            hbd = descriptors['hbd']
            hba = descriptors['hba']
            tpsa = descriptors['tpsa']
            
            # Lipinski's Rule of Five compliance
            lipinski_violations = 0
            if mw > 500: lipinski_violations += 1
            if logp > 5: lipinski_violations += 1
            if hbd > 5: lipinski_violations += 1
            if hba > 10: lipinski_violations += 1
            
            # TPSA constraint
            if tpsa > 140: lipinski_violations += 1
            
            drug_likeness = 5 - lipinski_violations  # Max score is 5
            return drug_likeness
    
    def optimize(self, initial_population, generations=100):
        """Run multi-objective optimization."""
        
        current_population = initial_population.copy()
        optimization_history = []
        
        for generation in range(generations):
            # Evaluate current population
            objective_values = self.evaluate_population(current_population)
            
            # Perform NSGA-II selection and breeding
            next_population = self.optimizer.evolve_population(
                population=current_population,
                objective_values=objective_values
            )
            
            # Analyze Pareto front
            pareto_indices = self.pareto_analyzer.find_pareto_front(objective_values)
            pareto_molecules = [current_population[i] for i in pareto_indices]
            pareto_objectives = objective_values[pareto_indices]
            
            # Calculate hypervolume (measure of Pareto front quality)
            hypervolume = self.pareto_analyzer.calculate_hypervolume(
                pareto_objectives,
                reference_point=np.zeros(len(self.objectives))
            )
            
            optimization_history.append({
                'generation': generation,
                'population_size': len(current_population),
                'pareto_size': len(pareto_molecules),
                'hypervolume': hypervolume,
                'pareto_molecules': pareto_molecules,
                'pareto_objectives': pareto_objectives
            })
            
            current_population = next_population
            
            # Log progress
            if generation % 10 == 0:
                print(f"Generation {generation}: Pareto size = {len(pareto_molecules)}, "
                      f"Hypervolume = {hypervolume:.3f}")
        
        # Final evaluation
        final_objectives = self.evaluate_population(current_population)
        final_pareto_indices = self.pareto_analyzer.find_pareto_front(final_objectives)
        final_pareto_molecules = [current_population[i] for i in final_pareto_indices]
        final_pareto_objectives = final_objectives[final_pareto_indices]
        
        return {
            'pareto_molecules': final_pareto_molecules,
            'pareto_objectives': final_pareto_objectives,
            'optimization_history': optimization_history,
            'final_population': current_population
        }

# Example: Multi-objective drug optimization
# Objectives: maximize drug-likeness, minimize toxicity, maximize novelty

drug_objectives = [
    {
        'name': 'drug_likeness',
        'type': 'drug_likeness',
        'descriptors': ['molecular_weight', 'logp', 'hbd', 'hba', 'tpsa'],
        'parameters': {}
    },
    {
        'name': 'low_toxicity',
        'type': 'target_value',
        'descriptors': ['mutagenicity_score'],
        'parameters': {
            'descriptor': 'mutagenicity_score',
            'target': 0.0,  # Lower is better
            'weight': 2.0
        }
    },
    {
        'name': 'synthetic_accessibility',
        'type': 'target_value',
        'descriptors': ['sa_score'],
        'parameters': {
            'descriptor': 'sa_score',
            'target': 2.0,  # SA score of 2 is reasonable
            'weight': 1.5
        }
    },
    {
        'name': 'activity_target',
        'type': 'range_satisfaction',
        'descriptors': ['molecular_weight', 'logp', 'tpsa'],
        'parameters': {
            'constraints': [
                {'descriptor': 'molecular_weight', 'min': 200, 'max': 500, 'weight': 1.0},
                {'descriptor': 'logp', 'min': 1.0, 'max': 4.0, 'weight': 1.0},
                {'descriptor': 'tpsa', 'min': 40, 'max': 120, 'weight': 1.0}
            ]
        }
    }
]

# Initialize multi-objective optimizer
mo_optimizer = MolecularMultiObjectiveOptimization(
    objectives=drug_objectives,
    population_size=200
)

# Starting population (drug-like molecules)
starting_molecules = load_drugbank_molecules()[:500]  # Use subset as starting population

# Optimize
optimization_result = mo_optimizer.optimize(
    initial_population=starting_molecules,
    generations=150
)

print(f"Found {len(optimization_result['pareto_molecules'])} Pareto-optimal molecules")

# Analyze Pareto front
pareto_df = pd.DataFrame({
    'molecule': optimization_result['pareto_molecules'],
    'drug_likeness': optimization_result['pareto_objectives'][:, 0],
    'low_toxicity': optimization_result['pareto_objectives'][:, 1],
    'synthetic_accessibility': optimization_result['pareto_objectives'][:, 2],
    'activity_target': optimization_result['pareto_objectives'][:, 3]
})

print("\nTop 5 Pareto-optimal molecules:")
print(pareto_df.head())
```

## Best Practices

### 1. Model Validation
```python
# Always validate AI models with held-out test sets
def validate_ai_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"Test MSE: {mse:.4f}")
    print(f"Test R²: {r2:.4f}")
    
    return mse < 0.1 and r2 > 0.8  # Acceptable thresholds
```

### 2. Chemical Validity Checking
```python
# Always validate generated molecules
def validate_generated_molecules(molecules):
    valid_molecules = []
    for mol in molecules:
        if is_valid_smiles(mol) and passes_chemical_filters(mol):
            valid_molecules.append(mol)
    
    validity_rate = len(valid_molecules) / len(molecules)
    print(f"Validity rate: {validity_rate:.2%}")
    
    return valid_molecules
```

### 3. Uncertainty Quantification
```python
# Always quantify prediction uncertainty
def predict_with_uncertainty(model, X):
    mean_pred = model.predict(X)
    std_pred = model.predict_std(X)  # If supported
    
    # Flag high-uncertainty predictions
    high_uncertainty = std_pred > 0.5
    print(f"High uncertainty predictions: {high_uncertainty.sum()}")
    
    return mean_pred, std_pred
```

---

*For specific AI applications in polymer and materials science, see the respective specialized guides.*
