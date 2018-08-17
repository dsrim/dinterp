# build dinterp module for python written in fortran using f2py

OBJECTS_DISTERP = dinterpm.o 
SOURCES_DISTERP = dinterpm.f90 
MODULES_DISTERP = dinterpm.mod

FC = gfortran
FFLAGS = -fPIC -fopenmp -O2
F2PY = f2py
LFLAGS = -lgomp

#F2PYLFLAGS = --f77flags="-fopenmp" --f90flags="-fopenmp" 

.PHONY: clean .so obj

obj: $(MODULES_DISTERP) $(OBJECTS_DISTERP) 

.so: _dinterp.so

_dinterp.so: $(MODULES_DISTERP) $(OBJECTS_DISTERP) _dinterp.pyf
	env FFLAGS='-fPIC -fopenmp -O2' f2py -c --f90flags="-O2" \
               -m _dinterp _dinterp.pyf $(OBJECTS_DISTERP)

%.o: %.f90 
	$(FC) $(FFLAGS) $(LFLAGS) -c $< -o $@

%.o: %.f 
	$(FC) $(FFLAGS) $(LFLAGS) -c $< -o $@

%.mod: %.f90
	$(FC) $(FFLAGS) $(LFLAGS) -c $< 

_dinterp.pyf:
	f2py -m _dinterp -h _dinterp.pyf $(SOURCES_DISTERP) 

clean:
	rm -f $(OBJECTS_DISTERP) $(MODULES_DISTERP) \
		  _dinterp.so _dinterp.pyf 

