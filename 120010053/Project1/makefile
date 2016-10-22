FILES := LC_Tank_Report.pdf
AUXFIlES := $(FILES:.pdf=.aux)
LOGFILES := $(FILES:.pdf=.log)

all: $(FILES)
%.pdf:	%.tex
		pdflatex $<

params:
		python3 LC_
	
.PHONY: clean
clean:
	$(RM) $(FILES) $(AUXFILES) $(LOGFILES)
