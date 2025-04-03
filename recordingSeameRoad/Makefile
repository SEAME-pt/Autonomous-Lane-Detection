CC = g++

# Flags de compilação com includes locais + OpenCV
CFLAGS = -Wall -Wextra -Werror -std=c++17 -O3 -g -Iincludes `pkg-config --cflags opencv4`

# Flags de link com SDL, i2c, pthread e OpenCV
LDFLAGS = -lSDL2 -lSDL2_image -lSDL2_ttf -lSDL2_mixer -li2c -lpthread `pkg-config --libs opencv4`

SRC = $(wildcard sources/*.cpp)
OBJ = $(SRC:.cpp=.o)
EXEC = jetracer

all: $(EXEC)

$(EXEC): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(EXEC)

run: $(EXEC)
	./$(EXEC)

re: clean all

.PHONY: all clean run

