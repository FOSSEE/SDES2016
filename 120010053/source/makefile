FILES := 120010053.pdf
TEXFILES := $(FILES:.pdf=.tex)
AUXFILES := $(TEXFILES:.tex=.aux)
LOGFILES := $(FILES:.pdf=.log)
OUTFILES := $(FILES:.pdf=.out)
BBL := $(FILES:.pdf=.bbl)
BLG := $(FILES:.pdf=.blg)

all: $(FILES)

%.pdf:	%.tex *.png *.sty
	\pdflatex $<
	bibtex $(AUXFILES)
	pdflatex $<
	pdflatex $<
		
	mkdir ../output/		
	mv -t ../output/ $(FILES) *.png $(AUXFILES) $(LOGFILES) \
	$(OUTFILES) $(BBL) $(BLG) *.sty

		
%.png %.sty: LC_Tank.py
	python3 LC_Tank.py
		
#%.sty: LC_Tank.py
#		python3 LC_Tank.py
		
	
.PHONY: clean
clean:
	$(RM) -r $(FILES) *.png $(AUXFILES) $(LOGFILES) \
	$(OUTFILES) $(BBL) $(BLG) *.sty

