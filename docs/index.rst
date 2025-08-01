ChemiAccel Documentation
========================

.. image:: _static/logo.png
   :alt: ChemiAccel Logo
   :align: center
   :scale: 50%

Welcome to ChemiAccel, a next-generation computational science platform designed to accelerate chemical and materials discovery through massively parallel, GPU-accelerated molecular descriptor calculations.

ChemiAccel provides 100-1000x speedups over traditional CPU-based tools like RDKit and Mordred, with first-class support for:

* **GPU-accelerated molecular descriptors** with CUDA, JAX, and RAPIDS backends
* **Polymer science** with specialized modules for copolymers and thermosets
* **Materials science** with crystal structure analysis and periodic descriptors
* **AI-driven discovery** with active learning, generative chemistry, and optimization
* **Cloud-native workflows** with native ML platform integration

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/core_concepts
   user_guide/backends
   user_guide/descriptors
   user_guide/cloud_deployment
   user_guide/hpc_usage

.. toctree::
   :maxdepth: 2
   :caption: Domain-Specific Guides

   domains/polymer_science
   domains/materials_science
   domains/ai_workflows
   domains/drug_discovery

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/core
   api/descriptors
   api/backends
   api/polymer
   api/materials
   api/ai
   api/cloud

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/basic_usage
   examples/advanced_features
   auto_examples/index

.. toctree::
   :maxdepth: 2
   :caption: Jupyter Notebooks

   notebooks/getting_started
   notebooks/polymer_analysis
   notebooks/materials_discovery
   notebooks/ai_workflows
   notebooks/cloud_deployment

.. toctree::
   :maxdepth: 2
   :caption: Development

   development/contributing
   development/testing
   development/benchmarking

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources

   changelog
   faq
   troubleshooting
   performance_tips

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`