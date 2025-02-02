.. _contributing:

Contributing
============

Start by installing the development dependencies:

.. code-block:: bash

  pip install tox black flake8


Then this package itself

.. code-block:: bash

  git clone https://github.com/lab-cosmo/scikit-matter
  cd scikit-matter
  pip install -e .

This install the package in development mode, making it importable globally
and allowing you to edit the code and directly use the updated version.

Running the tests
#################

.. code-block:: bash

  cd <scikit-matter PATH>
  # run unit tests
  tox
  # run the code formatter
  black --check .
  # run the linter
  flake8

You may want to setup your editor to automatically apply the
`black <https://black.readthedocs.io/en/stable/>`_ code formatter when saving your
files, there are plugins to do this with `all major
editors <https://black.readthedocs.io/en/stable/editor_integration.html>`_.


Issues and Pull Requests
########################

Having a problem with scikit-matter? Please let us know by `submitting an issue <https://github.com/lab-cosmo/scikit-matter/issues>`_.

Submit new features or bug fixes through a `pull request <https://github.com/lab-cosmo/scikit-matter/pulls>`_.


Contributing Datasets
#####################

Have an example dataset that would fit into scikit-matter?

Contributing a dataset is easy. First, copy your numpy file into
``skmatter/datasets/data/`` with an informative name. Here, we'll call it ``my-dataset.npz``.

Next, create a documentation file in ``skmatter/datasets/data/my-dataset.rst``.
This file should look like this:

.. code-block::

  .. _my-dataset:

  My Dataset
  ##########

  This is a summary of my dataset. My dataset was originally published in My Paper.

  Function Call
  -------------

  .. function:: skmatter.datasets.load_my_dataset

  Data Set Characteristics
  ------------------------

  :Number of Instances: ______

  :Number of Features: ______

  The representations were computed using the _____ package using the hyperparameters:


  +------------------------+------------+
  | key                    |   value    |
  +------------------------+------------+
  | hyperparameter 1       |    _____   |
  +------------------------+------------+
  | hyperparameter 2       |    _____   |
  +------------------------+------------+

  Of the ____ resulting features, ____ were selected via _____.

  References
  ----------

  Reference Code
  --------------


Then, show ``scikit-matter`` how to load your data by adding a loader function to
``skmatter/datasets/_base.py``. It should look like this:

.. code-block:: python

  def load_my_dataset():
      """Load and returns my dataset.

      Returns
      -------
      my_data : sklearn.utils.Bunch
          Dictionary-like object, with the following attributes:

          data : `sklearn.utils.Bunch` --
          contains the keys ``X`` and ``y``.
          My input vectors and properties, respectively.

          DESCR: `str` --
          The full description of the dataset.
      """
      module_path = dirname(__file__)
      target_filename = join(module_path, "data", "my-dataset.npz")
      raw_data = np.load(target_filename)
      data = Bunch(
          X=raw_data["X"],
          y=raw_data["y"],
      )
      with open(join(module_path, "descr", "my-dataset.rst")) as rst_file:
          fdescr = rst_file.read()

      return Bunch(data=data, DESCR=fdescr)

Add this function to ``skmatter/datasets/__init__.py``.

Finally, add a test to ``skmatter/tests/test_datasets.py`` to see that your dataset
loads properly. It should look something like this:

.. code-block:: python

  class MyDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.my_data = load_my_data()

    def test_load_my_data(self):
        # test if representations and properties have commensurate shape
        self.assertTrue(self.my_data.data.X.shape[0] == self.my_data.data.y.shape[0])

    def test_load_my_data_descr(self):
        self.my_data.DESCR


You're good to go! Time to submit a `pull request. <https://github.com/lab-cosmo/scikit-matter/pulls>`_


License
#######

This project is distributed under the BSD-3-Clauses license. By contributing to
it you agree to distribute your changes under the same license.
