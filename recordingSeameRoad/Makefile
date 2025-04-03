CC = g++
CFLAGS = -Wall -Wextra -Werror -std=c++17 -O3 -g -Iincludes `pkg-config --cflags opencv4`
LDFLAGS = -lSDL2 -li2c -lpthread `pkg-config --libs opencv4`

# Fontes
SRC_CONTROL = sources/main_control.cpp sources/jetracer.cpp
SRC_VIDEO   = sources/main_video.cpp

# Objetos
OBJ_CONTROL = $(SRC_CONTROL:.cpp=.o)
OBJ_VIDEO   = $(SRC_VIDEO:.cpp=.o)

# Executáveis
EXEC_CONTROL = jetracer_control
EXEC_VIDEO   = jetracer_video

all: $(EXEC_CONTROL) $(EXEC_VIDEO)

# Compilação dos programas individualmente
$(EXEC_CONTROL): $(OBJ_CONTROL)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(EXEC_VIDEO): $(OBJ_VIDEO)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Regra genérica de compilação
%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

# Limpa arquivos gerados
clean:
	rm -f $(OBJ_CONTROL) $(OBJ_VIDEO) $(EXEC_CONTROL) $(EXEC_VIDEO)

# Recompila tudo do zero
re: clean all

# Alvos específicos
control: $(EXEC_CONTROL)
video: $(EXEC_VIDEO)

# Executa os dois ao mesmo tempo
run: all
	@echo "Iniciando controle do JetRacer e gravação de vídeo..."
	@./$(EXEC_CONTROL) & ./$(EXEC_VIDEO)

.PHONY: all clean re control video run

