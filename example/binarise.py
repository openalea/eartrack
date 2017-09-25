from openalea.core.alea import load_package_manager, function
import alinea.phenomenal.phenomenal_config as pconf

pm = load_package_manager()
node_factory = pm['alinea.phenomenal.macros']['side_binarisation']
node_factory2 = pm['alinea.phenomenal.macros']['top_binarisation']
side_binarisation = function(node_factory)
top_binarisation = function(node_factory2)

p = pconf.getconfig()
opts_sv = pconf.sidebinarisation_options(p).values()
opts_tv = pconf.topbinarisation_options(p).values()


def side_bin(file):
    return side_binarisation(file, *opts_sv)[0]
    
    
res = top_binarisation(file, *opts_tv)[0]