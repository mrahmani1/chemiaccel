AI-Driven Discovery Workflows
=============================

This guide demonstrates advanced AI workflows for molecular discovery, optimization, and autonomous experimentation using ChemiAccel's AI modules.

Overview
--------

ChemiAccel's AI capabilities enable:

* **Inverse molecular design** - Generate molecules with target properties
* **Active learning** - Iteratively improve models with strategic sampling
* **Multi-objective optimization** - Balance competing molecular properties
* **Generative chemistry** - Create novel molecular structures
* **Autonomous experimentation** - AI-driven experimental planning and execution
* **Uncertainty quantification** - Assess prediction confidence

Core Workflow Patterns
-----------------------

Design-Make-Test-Analyze (DMTA) Cycle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The modern drug discovery cycle enhanced with AI:

.. code-block:: text

    [Design: AI Generation] → [Make: Synthesis Planning] → [Test: Automated Assays] → [Analyze: ML Analysis] → [Iterate]

Reinforcement Learning Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Continuous improvement through reward-based learning:

.. code-block:: text

    [State: Molecular Properties] → [Action: Structural Modifications] → [Reward: Property Improvement] → [Policy Update]

Multi-Agent Discovery
~~~~~~~~~~~~~~~~~~~~~

Collaborative AI agents for complex discovery tasks:

.. code-block:: text

    [Generator Agent] ↔ [Evaluator Agent] ↔ [Synthesizability Agent] ↔ [Experimental Agent]

Workflow 1: Inverse Molecular Design
-------------------------------------

Designing molecules with specific target properties.

Basic Inverse Design
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from chemi_accel.ai.inverse_design import MolecularInverseDesign
    from chemi_accel import Calculator

    # Initialize inverse design engine
    designer = MolecularInverseDesign(
        algorithm='genetic_algorithm',
        population_size=100,
        mutation_rate=0.1,
        crossover_rate=0.7
    )

    # Define target properties
    target_properties = {
        'molecular_weight': 350.0,     # Moderate size
        'logp': 2.5,                   # Good lipophilicity
        'tpsa': 60.0,                  # Good permeability
        'qed': 0.8,                    # High drug-likeness
        'synthetic_accessibility': 3.0  # Synthesizable
    }

    # Set property weights and tolerances
    property_weights = {
        'molecular_weight': 1.0,
        'logp': 2.0,           # High importance
        'tpsa': 2.0,           # High importance
        'qed': 3.0,            # Very high importance
        'synthetic_accessibility': 1.5
    }

    # Generate molecular candidates
    candidates = designer.generate_molecules(
        target_properties=target_properties,
        property_weights=property_weights,
        num_generations=50,
        convergence_threshold=0.95
    )

    # Analyze results
    print(f"Generated {len(candidates)} candidates")
    for i, mol in enumerate(candidates[:5]):
        print(f"\nCandidate {i+1}:")
        print(f"  SMILES: {mol['smiles']}")
        print(f"  MW: {mol['properties']['molecular_weight']:.1f}")
        print(f"  LogP: {mol['properties']['logp']:.2f}")
        print(f"  TPSA: {mol['properties']['tpsa']:.1f}")
        print(f"  QED: {mol['properties']['qed']:.3f}")
        print(f"  Fitness: {mol['fitness']:.3f}")

Advanced Multi-Objective Design
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from chemi_accel.ai.multi_objective_optimization import MultiObjectiveOptimizer

    # Multi-objective optimization with Pareto front
    optimizer = MultiObjectiveOptimizer(algorithm='nsga2')

    # Define competing objectives
    objectives = [
        {'name': 'potency', 'target': 'maximize', 'weight': 3.0},
        {'name': 'selectivity', 'target': 'maximize', 'weight': 2.0},
        {'name': 'toxicity', 'target': 'minimize', 'weight': 2.0},
        {'name': 'synthetic_complexity', 'target': 'minimize', 'weight': 1.0}
    ]

    # Run multi-objective optimization
    pareto_solutions = optimizer.optimize(
        objectives=objectives,
        population_size=200,
        generations=100,
        mutation_probability=0.15
    )

    # Analyze Pareto front
    print(f"Pareto front contains {len(pareto_solutions)} solutions")
    
    # Select balanced solution
    balanced_solution = optimizer.select_balanced_solution(
        pareto_solutions,
        preference_weights={'potency': 0.4, 'selectivity': 0.3, 'toxicity': 0.2, 'complexity': 0.1}
    )
    
    print(f"Balanced solution: {balanced_solution['smiles']}")

Workflow 2: Active Learning Pipeline
-------------------------------------

Iteratively improve models by strategically selecting new training data.

Basic Active Learning Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from chemi_accel.ai.active_learning import ActiveLearner
    from chemi_accel import Calculator
    import pandas as pd

    # Initialize active learner
    learner = ActiveLearner(
        base_model='random_forest',
        acquisition_function='expected_improvement',
        uncertainty_method='ensemble'
    )

    # Initial training data
    initial_smiles = [
        "CCO", "CCN", "CC(=O)O", "c1ccccc1", "CCc1ccccc1"
    ]
    initial_targets = [0.2, 0.4, 0.1, 0.8, 0.9]  # Target activity

    # Train initial model
    learner.fit(initial_smiles, initial_targets)

    # Candidate pool for selection
    candidate_pool = pd.read_csv('candidate_molecules.csv')['smiles'].tolist()

    # Active learning loop
    for iteration in range(10):
        print(f"\n=== Active Learning Iteration {iteration + 1} ===")
        
        # Select next batch of molecules to evaluate
        next_molecules = learner.select_next_batch(
            candidate_pool=candidate_pool,
            batch_size=20,
            diversity_factor=0.3  # Balance exploitation vs exploration
        )
        
        print(f"Selected {len(next_molecules)} molecules for evaluation")
        
        # Simulate experimental evaluation
        # In practice, these would be real experimental measurements
        new_targets = simulate_experimental_evaluation(next_molecules)
        
        # Update model with new data
        learner.update(next_molecules, new_targets)
        
        # Remove evaluated molecules from candidate pool
        candidate_pool = [mol for mol in candidate_pool if mol not in next_molecules]
        
        # Evaluate model performance
        performance = learner.evaluate_performance()
        print(f"Model R²: {performance['r2']:.3f}")
        print(f"Model RMSE: {performance['rmse']:.3f}")
        
        # Check convergence
        if performance['r2'] > 0.95:
            print("Convergence achieved!")
            break

    # Final model predictions
    final_predictions = learner.predict(candidate_pool[:100])
    print(f"Final predictions range: {min(final_predictions):.3f} - {max(final_predictions):.3f}")

Advanced Active Learning with Uncertainty
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from chemi_accel.ai.uncertainty_quantification import UncertaintyQuantifier

    # Enhanced active learning with uncertainty quantification
    uq_learner = UncertaintyQuantifier(
        method='monte_carlo_dropout',
        n_samples=100,
        dropout_rate=0.1
    )

    def active_learning_with_uncertainty(initial_data, candidate_pool, budget=200):
        """Active learning with sophisticated uncertainty quantification"""
        
        # Initialize model
        uq_learner.fit(initial_data['smiles'], initial_data['targets'])
        
        selected_molecules = []
        measured_targets = []
        
        for step in range(budget // 20):  # 20 molecules per step
            print(f"\nActive Learning Step {step + 1}")
            
            # Get predictions with uncertainty for all candidates
            predictions = uq_learner.predict_with_uncertainty(candidate_pool)
            
            # Calculate acquisition scores
            uncertainty_scores = [pred['epistemic_uncertainty'] for pred in predictions]
            expected_values = [pred['mean'] for pred in predictions]
            
            # Multi-criteria selection
            acquisition_scores = []
            for i, (uncertainty, expected_value) in enumerate(zip(uncertainty_scores, expected_values)):
                # Balance uncertainty and expected improvement
                acquisition_score = (
                    0.6 * uncertainty +  # Exploration
                    0.4 * max(0, expected_value - 0.7)  # Exploitation (target > 0.7)
                )
                acquisition_scores.append((acquisition_score, i))
            
            # Select top candidates
            acquisition_scores.sort(reverse=True)
            selected_indices = [idx for _, idx in acquisition_scores[:20]]
            selected_smiles = [candidate_pool[i] for i in selected_indices]
            
            # Simulate evaluation
            new_measurements = simulate_experimental_evaluation(selected_smiles)
            
            # Update model
            selected_molecules.extend(selected_smiles)
            measured_targets.extend(new_measurements)
            
            # Retrain with all data
            all_smiles = initial_data['smiles'] + selected_molecules
            all_targets = initial_data['targets'] + measured_targets
            uq_learner.fit(all_smiles, all_targets)
            
            # Remove selected molecules from pool
            candidate_pool = [mol for i, mol in enumerate(candidate_pool) if i not in selected_indices]
            
            # Evaluate progress
            test_predictions = uq_learner.predict_with_uncertainty(candidate_pool[:100])
            avg_uncertainty = sum(pred['total_uncertainty'] for pred in test_predictions) / len(test_predictions)
            
            print(f"Selected molecules: {len(selected_smiles)}")
            print(f"Average uncertainty in remaining pool: {avg_uncertainty:.3f}")
            
        return selected_molecules, measured_targets

Workflow 3: Generative Chemistry
---------------------------------

Create novel molecular structures using deep generative models.

Transformer-based Molecular Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from chemi_accel.ai.generative_chemistry import TransformerGenerator
    from chemi_accel import Calculator

    # Initialize transformer-based generator
    generator = TransformerGenerator(
        model_type='gpt_molecular',
        vocab_size=50,
        hidden_size=512,
        n_layers=8,
        n_heads=8
    )

    # Training data (large dataset of SMILES)
    training_smiles = pd.read_csv('chembl_molecules.csv')['smiles'].tolist()

    # Train generative model
    print("Training molecular transformer...")
    generator.train(
        training_data=training_smiles,
        epochs=100,
        batch_size=64,
        learning_rate=1e-4,
        validation_split=0.1
    )

    # Generate novel molecules
    novel_molecules = generator.generate(
        num_molecules=10000,
        temperature=0.8,          # Control creativity
        max_length=100,          # Maximum SMILES length
        filter_invalid=True,     # Remove invalid SMILES
        diversity_penalty=0.1    # Encourage diversity
    )

    print(f"Generated {len(novel_molecules)} novel molecules")

    # Post-process and filter
    calc = Calculator()
    
    # Calculate properties for generated molecules
    properties = calc.pandas(novel_molecules, descriptors=[
        'molecular_weight', 'logp', 'tpsa', 'qed'
    ])

    # Apply drug-likeness filters
    drug_like = properties[
        (properties['molecular_weight'] >= 150) &
        (properties['molecular_weight'] <= 500) &
        (properties['logp'] >= -0.4) &
        (properties['logp'] <= 5.6) &
        (properties['tpsa'] <= 200) &
        (properties['qed'] >= 0.5)
    ]

    print(f"Drug-like molecules: {len(drug_like)}")

Conditional Generation
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from chemi_accel.ai.generative_chemistry import ConditionalGenerator

    # Generate molecules with specific properties
    conditional_gen = ConditionalGenerator(model_type='cvae')

    # Train with property conditioning
    training_data = pd.read_csv('training_molecules.csv')
    conditional_gen.train(
        smiles=training_data['smiles'],
        properties=training_data[['molecular_weight', 'logp', 'tpsa']],
        epochs=200
    )

    # Generate molecules with target properties
    target_conditions = {
        'molecular_weight': 300.0,
        'logp': 2.0,
        'tpsa': 80.0
    }

    conditional_molecules = conditional_gen.generate_conditional(
        conditions=target_conditions,
        num_molecules=1000,
        tolerance={'molecular_weight': 50, 'logp': 1.0, 'tpsa': 20}
    )

    print(f"Generated {len(conditional_molecules)} molecules with target properties")

Workflow 4: Autonomous Laboratory Integration
---------------------------------------------

AI-driven experimental planning, execution, and analysis.

Autonomous Experimental Planning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from chemi_accel.ai.autonomous_lab import AutonomousLab
    from chemi_accel.ai.experimental_optimization import ExperimentalOptimizer

    # Initialize autonomous laboratory
    lab = AutonomousLab(
        equipment_config={
            'synthesizers': ['microwave_1', 'microwave_2'],
            'purification': ['prep_hplc', 'flash_column'],
            'analysis': ['lc_ms', 'nmr_400', 'hplc']
        },
        resource_constraints={
            'budget': 50000,           # USD
            'time_limit': 168,         # Hours (1 week)
            'material_limits': {       # Available materials
                'solvents': ['DMF', 'DMSO', 'EtOH', 'MeCN'],
                'reagents': ['NaH', 'K2CO3', 'DIPEA'],
                'starting_materials': 100  # Number available
            }
        }
    )

    # Define experimental objective
    optimization_target = {
        'property': 'binding_affinity',
        'target_value': 10.0,  # nM
        'optimization_direction': 'minimize'
    }

    # Plan autonomous experimental campaign
    campaign = lab.plan_experimental_campaign(
        candidate_molecules=next_molecules,
        optimization_target=optimization_target,
        campaign_duration_days=7,
        max_experiments=50
    )

    print(f"Planned {len(campaign['experiments'])} experiments")
    print(f"Estimated cost: ${campaign['estimated_cost']:,.2f}")
    print(f"Expected completion: {campaign['estimated_duration']} hours")

    # Execute experiments autonomously
    results = lab.execute_campaign(campaign)

    # Analyze results and plan next iteration
    analysis = lab.analyze_campaign_results(results)
    print(f"Success rate: {analysis['success_rate']:.1%}")
    print(f"Best result: {analysis['best_result']['value']:.2f} nM")

Reinforcement Learning Process Control
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from chemi_accel.ai.rl_process_control import RLProcessController

    # Initialize RL controller for reaction optimization
    rl_controller = RLProcessController(
        algorithm='ppo',
        environment='batch_reactor'
    )

    # Define process parameters
    process_space = {
        'temperature': (50, 200),      # Celsius
        'pressure': (1, 10),           # bar
        'flow_rate': (0.1, 5.0),       # mL/min
        'catalyst_loading': (0.01, 0.1), # mol%
        'reaction_time': (30, 300)     # minutes
    }

    # Define reward function
    def reaction_reward(conditions, outcome):
        """Calculate reward based on yield, selectivity, and cost"""
        yield_score = outcome['yield'] / 100.0  # Normalize to 0-1
        selectivity_score = outcome['selectivity'] / 100.0
        
        # Cost penalty (higher temperature/pressure = higher cost)
        cost_penalty = (
            (conditions['temperature'] - 50) / 150 * 0.1 +
            (conditions['pressure'] - 1) / 9 * 0.1
        )
        
        return 0.5 * yield_score + 0.3 * selectivity_score - cost_penalty

    # Train RL agent
    rl_controller.train(
        process_space=process_space,
        reward_function=reaction_reward,
        episodes=1000,
        max_steps_per_episode=50
    )

    # Deploy optimized controller
    optimal_conditions = rl_controller.optimize_reaction(
        target_yield=0.90,
        target_selectivity=0.95,
        max_cost_per_reaction=25.0
    )

    print("Optimal reaction conditions:")
    for param, value in optimal_conditions.items():
        print(f"  {param}: {value}")

Workflow 5: Integrated Discovery Pipeline
------------------------------------------

Complete end-to-end AI-driven discovery workflow.

Full Pipeline Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    class IntegratedDiscoveryPipeline:
        """Complete AI-driven molecular discovery pipeline"""
        
        def __init__(self):
            self.generator = TransformerGenerator()
            self.inverse_designer = MolecularInverseDesign()
            self.active_learner = ActiveLearner()
            self.autonomous_lab = AutonomousLab()
            self.calculator = Calculator(backend='cuda')
            
        def run_discovery_campaign(self, target_profile, budget=100000, timeline_days=30):
            """Run complete discovery campaign"""
            
            print("=== PHASE 1: MOLECULAR GENERATION ===")
            # Generate initial molecule library
            initial_library = self.generator.generate(
                num_molecules=50000,
                temperature=0.7,
                filter_invalid=True
            )
            
            # Calculate descriptors for screening
            library_props = self.calculator.pandas(
                initial_library,
                descriptors=['molecular_weight', 'logp', 'tpsa', 'qed']
            )
            
            # Pre-filter library
            filtered_library = self._apply_initial_filters(library_props)
            print(f"Initial library: {len(filtered_library)} molecules")
            
            print("\n=== PHASE 2: INVERSE DESIGN OPTIMIZATION ===")
            # Refine using inverse design
            optimized_molecules = self.inverse_designer.generate_molecules(
                target_properties=target_profile,
                initial_population=filtered_library[:1000],
                num_generations=50
            )
            
            print(f"Optimized candidates: {len(optimized_molecules)}")
            
            print("\n=== PHASE 3: ACTIVE LEARNING SELECTION ===")
            # Use active learning to select best candidates for synthesis
            # Start with small experimental set
            initial_experimental_set = optimized_molecules[:20]
            
            # Simulate initial experiments
            initial_results = self._simulate_experiments(initial_experimental_set)
            
            # Train active learning model
            self.active_learner.fit(
                [mol['smiles'] for mol in initial_experimental_set],
                [result['activity'] for result in initial_results]
            )
            
            # Iteratively select and evaluate candidates
            all_experimental_results = initial_results.copy()
            remaining_candidates = optimized_molecules[20:]
            
            for iteration in range(10):  # 10 active learning iterations
                print(f"\nActive Learning Iteration {iteration + 1}")
                
                # Select next batch
                next_batch = self.active_learner.select_next_batch(
                    [mol['smiles'] for mol in remaining_candidates],
                    batch_size=10
                )
                
                # Plan and execute experiments
                experimental_plan = self.autonomous_lab.plan_experiments(
                    next_batch,
                    experimental_protocols=['binding_assay', 'cytotoxicity']
                )
                
                batch_results = self.autonomous_lab.execute_experiments(experimental_plan)
                all_experimental_results.extend(batch_results)
                
                # Update model
                self.active_learner.update(
                    next_batch,
                    [result['activity'] for result in batch_results]
                )
                
                # Remove tested molecules
                tested_smiles = set(next_batch)
                remaining_candidates = [
                    mol for mol in remaining_candidates 
                    if mol['smiles'] not in tested_smiles
                ]
                
                print(f"Tested: {len(all_experimental_results)} molecules")
                print(f"Remaining: {len(remaining_candidates)} candidates")
            
            print("\n=== PHASE 4: RESULTS ANALYSIS ===")
            # Analyze all results
            best_results = sorted(
                all_experimental_results,
                key=lambda x: x['activity'],
                reverse=True
            )[:10]
            
            print("Top 10 Candidates:")
            for i, result in enumerate(best_results):
                print(f"{i+1}. {result['smiles']}")
                print(f"   Activity: {result['activity']:.3f}")
                print(f"   Selectivity: {result.get('selectivity', 'N/A')}")
                print(f"   Cytotoxicity: {result.get('cytotoxicity', 'N/A')}")
            
            # Generate final report
            report = self._generate_discovery_report(
                all_experimental_results,
                target_profile,
                budget,
                timeline_days
            )
            
            return {
                'best_candidates': best_results,
                'all_results': all_experimental_results,
                'report': report
            }
        
        def _apply_initial_filters(self, library_props):
            """Apply drug-likeness and other initial filters"""
            return library_props[
                (library_props['molecular_weight'] >= 150) &
                (library_props['molecular_weight'] <= 500) &
                (library_props['logp'] >= -0.4) &
                (library_props['logp'] <= 5.6) &
                (library_props['qed'] >= 0.5)
            ]
        
        def _simulate_experiments(self, molecules):
            """Simulate experimental evaluation (replace with real experiments)"""
            import random
            results = []
            for mol in molecules:
                results.append({
                    'smiles': mol['smiles'],
                    'activity': random.uniform(0.1, 1.0),
                    'selectivity': random.uniform(0.5, 1.0),
                    'cytotoxicity': random.uniform(0.0, 0.3)
                })
            return results
        
        def _generate_discovery_report(self, results, target_profile, budget, timeline):
            """Generate comprehensive discovery report"""
            return {
                'campaign_summary': {
                    'molecules_tested': len(results),
                    'success_rate': len([r for r in results if r['activity'] > 0.7]) / len(results),
                    'best_activity': max(r['activity'] for r in results),
                    'budget_used': budget * 0.85,  # Simulated
                    'timeline_days': timeline
                },
                'target_achievement': self._assess_target_achievement(results, target_profile),
                'recommendations': self._generate_recommendations(results)
            }

    # Run integrated discovery campaign
    if __name__ == "__main__":
        pipeline = IntegratedDiscoveryPipeline()
        
        target_profile = {
            'molecular_weight': 350,
            'logp': 2.5,
            'tpsa': 70,
            'binding_affinity': 0.9  # Target activity
        }
        
        results = pipeline.run_discovery_campaign(
            target_profile=target_profile,
            budget=100000,
            timeline_days=30
        )
        
        print(f"\nDiscovery campaign completed successfully!")
        print(f"Best candidate: {results['best_candidates'][0]['smiles']}")
        print(f"Activity: {results['best_candidates'][0]['activity']:.3f}")

Best Practices for AI Workflows
--------------------------------

Model Validation
~~~~~~~~~~~~~~~~

1. **Cross-validation** - Use proper cross-validation for model assessment
2. **External validation** - Test on truly external datasets
3. **Temporal validation** - Validate on data from different time periods
4. **Prospective validation** - Test predictions with real experiments

Data Quality
~~~~~~~~~~~~

1. **Curation** - Implement robust data curation pipelines
2. **Standardization** - Standardize molecular representations
3. **Outlier detection** - Identify and handle outliers appropriately
4. **Bias assessment** - Check for dataset biases

Uncertainty Management
~~~~~~~~~~~~~~~~~~~~~~

1. **Quantify uncertainty** - Always provide uncertainty estimates
2. **Calibration** - Ensure uncertainty estimates are well-calibrated
3. **Decision making** - Use uncertainty in decision making processes
4. **Communication** - Clearly communicate uncertainty to stakeholders

Reproducibility
~~~~~~~~~~~~~~~

1. **Version control** - Track all code and model versions
2. **Seed management** - Set random seeds for reproducibility
3. **Environment documentation** - Document computational environments
4. **Data provenance** - Track data sources and transformations

For more detailed examples and advanced workflows, see the :doc:`../examples/advanced_features` and :doc:`../notebooks/ai_workflows` sections.