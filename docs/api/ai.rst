AI & Machine Learning API Reference
====================================

Advanced AI-driven discovery tools including active learning, generative chemistry, and multi-objective optimization.

Inverse Design
--------------

.. automodule:: chemi_accel.ai.inverse_design
    :members:
    :undoc-members:
    :show-inheritance:

Generative Chemistry
--------------------

.. automodule:: chemi_accel.ai.generative_chemistry
    :members:
    :undoc-members:
    :show-inheritance:

Active Learning
---------------

.. automodule:: chemi_accel.ai.active_learning
    :members:
    :undoc-members:
    :show-inheritance:

Multi-Objective Optimization
----------------------------

.. automodule:: chemi_accel.ai.multi_objective_optimization
    :members:
    :undoc-members:
    :show-inheritance:

Bayesian Optimization
---------------------

.. automodule:: chemi_accel.ai.bayesian_optimization
    :members:
    :undoc-members:
    :show-inheritance:

Uncertainty Quantification
--------------------------

.. automodule:: chemi_accel.ai.uncertainty_quantification
    :members:
    :undoc-members:
    :show-inheritance:

Autonomous Laboratory
---------------------

.. automodule:: chemi_accel.ai.autonomous_lab
    :members:
    :undoc-members:
    :show-inheritance:

Law Discovery
-------------

.. automodule:: chemi_accel.ai.law_discovery
    :members:
    :undoc-members:
    :show-inheritance:

Experimental Optimization
-------------------------

.. automodule:: chemi_accel.ai.experimental_optimization
    :members:
    :undoc-members:
    :show-inheritance:

Reinforcement Learning Process Control
--------------------------------------

.. automodule:: chemi_accel.ai.rl_process_control
    :members:
    :undoc-members:
    :show-inheritance:

QML Optimization
----------------

.. automodule:: chemi_accel.ai.qml_optimization
    :members:
    :undoc-members:
    :show-inheritance:

QML Inverse Design
------------------

.. automodule:: chemi_accel.ai.qml_inverse_design
    :members:
    :undoc-members:
    :show-inheritance:

Usage Examples
--------------

Molecular Inverse Design
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from chemi_accel.ai.inverse_design import MolecularInverseDesign

    designer = MolecularInverseDesign()
    
    # Define target properties
    target_properties = {
        'logp': 2.5,
        'tpsa': 60.0,
        'molecular_weight': 300.0,
        'qed': 0.8
    }
    
    # Generate molecular candidates
    candidates = designer.generate_molecules(
        target_properties=target_properties,
        num_generations=100,
        population_size=100,
        mutation_rate=0.1
    )
    
    # Display results
    for i, mol in enumerate(candidates[:5]):
        print(f"Candidate {i+1}:")
        print(f"  SMILES: {mol['smiles']}")
        print(f"  Predicted LogP: {mol['properties']['logp']:.2f}")
        print(f"  Fitness: {mol['fitness']:.3f}")

Generative Molecular Design
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from chemi_accel.ai.generative_chemistry import GenerativeChemistry

    generator = GenerativeChemistry(model_type='transformer')
    
    # Train on custom dataset
    training_smiles = ["CCO", "CCN", "CC(=O)O"]  # Example dataset
    generator.train(training_smiles, epochs=100)
    
    # Generate novel molecules
    novel_molecules = generator.generate(
        num_molecules=1000,
        temperature=0.8,
        diversity_filter=True
    )
    
    # Filter by drug-likeness
    drug_like = generator.filter_drug_like(novel_molecules)
    print(f"Generated {len(drug_like)} drug-like molecules")

Active Learning Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from chemi_accel.ai.active_learning import ActiveLearner

    learner = ActiveLearner(
        base_model='random_forest',
        acquisition_function='expected_improvement'
    )
    
    # Initial training set
    initial_smiles = ["CCO", "CCN", "CC(=O)O"]
    initial_targets = [0.5, 0.7, 0.3]  # Target property values
    
    # Start active learning loop
    learner.fit(initial_smiles, initial_targets)
    
    # Candidate pool for selection
    candidate_pool = ["CC(C)O", "CC(C)N", "CC(C)(C)O"]
    
    # Select next molecules to synthesize/test
    next_molecules = learner.select_next_batch(
        candidate_pool=candidate_pool,
        batch_size=5
    )
    
    print(f"Next molecules to evaluate: {next_molecules}")

Multi-Objective Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from chemi_accel.ai.multi_objective_optimization import MultiObjectiveOptimizer

    optimizer = MultiObjectiveOptimizer(algorithm='nsga2')
    
    # Define multiple objectives
    objectives = {
        'logp': {'target': 2.5, 'weight': 1.0, 'minimize': False},
        'tpsa': {'target': 60.0, 'weight': 1.0, 'minimize': True},
        'qed': {'target': 0.8, 'weight': 2.0, 'minimize': False}
    }
    
    # Run optimization
    pareto_front = optimizer.optimize(
        objectives=objectives,
        population_size=100,
        generations=50
    )
    
    # Analyze Pareto front
    print(f"Pareto front contains {len(pareto_front)} solutions")
    for i, solution in enumerate(pareto_front[:3]):
        print(f"Solution {i+1}:")
        print(f"  SMILES: {solution['smiles']}")
        print(f"  LogP: {solution['logp']:.2f}")
        print(f"  TPSA: {solution['tpsa']:.1f}")
        print(f"  QED: {solution['qed']:.3f}")

Bayesian Optimization
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from chemi_accel.ai.bayesian_optimization import BayesianOptimizer

    optimizer = BayesianOptimizer(
        surrogate_model='gaussian_process',
        acquisition_function='upper_confidence_bound'
    )
    
    # Define search space
    search_space = {
        'molecular_weight': (100, 500),
        'logp': (-2, 5),
        'tpsa': (0, 200)
    }
    
    # Optimize synthetic accessibility
    def objective_function(params):
        # Calculate synthetic accessibility score
        return calculate_sa_score(params)
    
    # Run optimization
    best_result = optimizer.optimize(
        objective_function=objective_function,
        search_space=search_space,
        n_iterations=50
    )
    
    print(f"Best parameters: {best_result['parameters']}")
    print(f"Best score: {best_result['score']:.3f}")

Uncertainty Quantification
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from chemi_accel.ai.uncertainty_quantification import UncertaintyQuantifier

    uq = UncertaintyQuantifier(method='monte_carlo_dropout')
    
    # Train model with uncertainty estimation
    smiles_train = ["CCO", "CCN", "CC(=O)O"]
    targets_train = [0.5, 0.7, 0.3]
    
    uq.fit(smiles_train, targets_train)
    
    # Make predictions with uncertainty
    test_smiles = ["CC(C)O", "CC(C)N"]
    predictions = uq.predict_with_uncertainty(test_smiles)
    
    for i, (smiles, pred) in enumerate(zip(test_smiles, predictions)):
        print(f"Molecule {i+1}: {smiles}")
        print(f"  Prediction: {pred['mean']:.3f} ± {pred['std']:.3f}")
        print(f"  Confidence: {pred['confidence']:.3f}")

Autonomous Laboratory Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from chemi_accel.ai.autonomous_lab import AutonomousLab

    lab = AutonomousLab()
    
    # Define experimental workflow
    workflow = {
        'synthesis': {'method': 'microwave', 'temperature': 120, 'time': 300},
        'purification': {'method': 'column_chromatography'},
        'characterization': ['nmr', 'lc_ms', 'hplc']
    }
    
    # Plan experiments
    experiment_plan = lab.plan_experiments(
        target_molecules=next_molecules,
        workflow=workflow,
        resource_constraints={'budget': 10000, 'time_limit': 168}  # 1 week
    )
    
    # Execute experiments
    results = lab.execute_experiments(experiment_plan)
    
    # Update model with new data
    learner.update(results['smiles'], results['measured_properties'])

Reinforcement Learning Process Control
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from chemi_accel.ai.rl_process_control import RLProcessController

    controller = RLProcessController(
        algorithm='ppo',
        environment='chemical_reactor'
    )
    
    # Define process parameters
    process_config = {
        'reactor_volume': 1.0,  # L
        'temperature_range': (20, 200),  # °C
        'pressure_range': (1, 10),  # bar
        'flow_rate_range': (0.1, 2.0)  # L/min
    }
    
    # Train RL agent
    controller.train(
        process_config=process_config,
        reward_function='yield_selectivity',
        episodes=1000
    )
    
    # Deploy for process control
    optimal_conditions = controller.optimize_process(
        target_yield=0.95,
        target_selectivity=0.90
    )
    
    print(f"Optimal conditions: {optimal_conditions}")

QML-Enhanced Molecular Design
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from chemi_accel.ai.qml_inverse_design import QMLInverseDesign

    qml_designer = QMLInverseDesign(qml_method='ani2x')
    
    # Target quantum properties
    target_properties = {
        'homo_lumo_gap': 3.5,    # eV
        'dipole_moment': 2.0,    # Debye
        'polarizability': 15.0,  # Å³
        'total_energy': -500.0   # Hartree
    }
    
    # Generate molecules with QML optimization
    qml_candidates = qml_designer.generate_molecules(
        target_properties=target_properties,
        qml_optimization=True,
        geometry_optimization=True
    )
    
    print(f"Generated {len(qml_candidates)} QML-optimized candidates")
    for candidate in qml_candidates[:3]:
        print(f"SMILES: {candidate['smiles']}")
        print(f"HOMO-LUMO gap: {candidate['homo_lumo_gap']:.2f} eV")
        print(f"Dipole moment: {candidate['dipole_moment']:.2f} D")