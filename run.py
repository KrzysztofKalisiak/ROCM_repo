from GeoGuesserSystem import *

BR = system_loader()

BR.train(100)
save_system(BR)