simplega

A Simple Genetic Algorithm solver in python.

The code runs on pure python and is not extremely optimized. It is easy to adapt.

Dependencies are very few for the base class. The SELFIES solver requires selfies and most functionalities for molecular exploration (chemistry module) require fairly modern rdkit.
Additional features require alive-progress (for progress bars, very useful) and functools for cacheing. However, these are implemented by monkeypatching the base class, and thus no functionality is lost without them.

Installation is as simple as:
python setup.py install --record files.txt

This ensures easy uninstall. Just remove all files listed in files.txt using:
rm $(cat files.txt)





