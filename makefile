
FILE=presentation

all:
	pdflatex $(FILE).tex
	pdflatex $(FILE).tex

clean:
	rm -f $(FILE).aux $(FILE).log $(FILE).nav $(FILE).out $(FILE).snm $(FILE).toc missfont.log

fclean: clean
	rm -f $(FILE).pdf

re: fclean all

