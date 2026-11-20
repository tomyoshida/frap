.. frappe documentation master file, created by
   sphinx-quickstart on Mon Jan 19 16:41:20 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=======================
FRAP documentation 🍹
=======================

Do you have observational data of the dust continuum emission from a protoplanetary disk?
**Flexible Radial Analysis of Protoplanetary disks (FRAP)** provides an easy-to-use tool to retrieve dust properties, such as the dust surface density distribution, in disks.

.. container:: center
   
   .. video:: _static/demo_top.mp4
      :width: 100%
      :autoplay:
      :loop:
      :muted:
      :playsinline:
      :nocontrols:
      :align: center


.. raw:: html

   <div style="text-align: center; font-style: italic; margin-top: 0px; margin-bottom: 20px;">
      Animation of the posterior sampling process for dust surface density with FRAP.
   </div>

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   introduction
   installation
   start

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorial_0
   tutorial_1
   tutorial_2

.. toctree::
   :maxdepth: 1
   :caption: Advanced tutorials

   adv_1
   adv_2
   adv_3
   adv_4
   adv_5

.. toctree::
   :maxdepth: 1
   :caption: Reference

   autoapi/index
   release_notes


Citation
--------

If you find FRAP useful for your research, please cite our paper:

Yoshida, T. C., Macías, E., Rosotti, G., Facchini, S., Viscardi, E.,
& Doi, K. (2026).
*Direct Retrieval of Protoplanetary Disk Dust Properties Using
Auto-differentiable Gaussian Processes and Their Application
to the HD 169142 Disk*.
The Astrophysical Journal, 1007, 198.

`SciX <https://scixplorer.org/abs/2026ApJ..1007..198Y>`_
| `DOI <https://doi.org/10.3847/1538-4357/ae89ad>`_


BibTeX
~~~~~~

.. code-block:: bibtex

   @ARTICLE{2026ApJ..1007..198Y,
     author = {{Yoshida}, Tomohiro C. and {Mac{\'\i}as}, Enrique and
               {Rosotti}, Giovanni and {Facchini}, Stefano and
               {Viscardi}, Elena and {Doi}, Kiyoaki},
     title = "{Direct Retrieval of Protoplanetary Disk Dust Properties Using Auto-differentiable Gaussian Processes and Their Application to the HD 169142 Disk}",
     journal = {The Astrophysical Journal},
     year = {2026},
     month = aug,
     volume = {1007},
     number = {2},
     pages = {198},
     doi = {10.3847/1538-4357/ae89ad},
     archivePrefix = {arXiv},
     eprint = {2607.12618},
     primaryClass = {astro-ph.EP},
     adsurl = {https://scixplorer.org/abs/2026ApJ..1007..198Y}
   }


Disclaimer
----------

FRAP is provided "as is", without warranty of any kind.
Users are responsible for validating their results and the
appropriateness of the underlying assumptions.


Contact
-------

`Tomohiro C. Yoshida <https://tomyoshida.github.io/>`_ (Università degli Studi di Milano) / tomohiroyoshida.astro [at] gmail.com