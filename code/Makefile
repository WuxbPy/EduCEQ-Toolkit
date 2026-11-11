PY ?= python3
RESULTDIR ?= ../results

.PHONY: all test tables logs clean

all: test tables

test:
	$(PY) -m pytest tests -q

tables: logs
	@echo "all evaluation tables regenerated in $(RESULTDIR)"

logs:
	@mkdir -p $(RESULTDIR)
	$(PY) calibration.py --reps 200      | tee $(RESULTDIR)/calibration.log
	$(PY) attenuation.py --reps 1000     | tee $(RESULTDIR)/attenuation.log
	$(PY) demo.py                        | tee $(RESULTDIR)/demo.log
	$(PY) examples/quickstart.py         | tee $(RESULTDIR)/quickstart.log
	$(PY) examples/us_panel.py           | tee $(RESULTDIR)/us_panel.log
	$(PY) evaluation/support_sensitivity.py  | tee $(RESULTDIR)/support_sensitivity.log
	$(PY) evaluation/accept_reject.py --reps 200 | tee $(RESULTDIR)/accept_reject.log
	$(PY) evaluation/weight_heterogeneity.py | tee $(RESULTDIR)/weight_heterogeneity.log
	$(PY) evaluation/pipeline_e2e.py     | tee $(RESULTDIR)/pipeline_e2e.log
	$(PY) evaluation/sfdep_parity.py     | tee $(RESULTDIR)/sfdep_parity.log
	$(PY) -m pytest tests -q             | tee $(RESULTDIR)/pytest.log
	$(PY) -c "import numpy, scipy, pytest; print('numpy==%s' % numpy.__version__); print('scipy==%s' % scipy.__version__); print('pytest==%s' % pytest.__version__)" > $(RESULTDIR)/environment.txt

clean:
	rm -f $(RESULTDIR)/*.log $(RESULTDIR)/*.csv
