# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Produce verbose output by default.
VERBOSE = 1

# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jason/school/cs569/programs/assignment2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jason/school/cs569/programs/assignment2/build

# Include any dependencies generated for this target.
include CMakeFiles/jacobi.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/jacobi.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/jacobi.dir/flags.make

CMakeFiles/jacobi.dir/jacobi.c.o: CMakeFiles/jacobi.dir/flags.make
CMakeFiles/jacobi.dir/jacobi.c.o: ../jacobi.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jason/school/cs569/programs/assignment2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/jacobi.dir/jacobi.c.o"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/jacobi.dir/jacobi.c.o   -c /home/jason/school/cs569/programs/assignment2/jacobi.c

CMakeFiles/jacobi.dir/jacobi.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/jacobi.dir/jacobi.c.i"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/jason/school/cs569/programs/assignment2/jacobi.c > CMakeFiles/jacobi.dir/jacobi.c.i

CMakeFiles/jacobi.dir/jacobi.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/jacobi.dir/jacobi.c.s"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/jason/school/cs569/programs/assignment2/jacobi.c -o CMakeFiles/jacobi.dir/jacobi.c.s

CMakeFiles/jacobi.dir/jacobi.c.o.requires:

.PHONY : CMakeFiles/jacobi.dir/jacobi.c.o.requires

CMakeFiles/jacobi.dir/jacobi.c.o.provides: CMakeFiles/jacobi.dir/jacobi.c.o.requires
	$(MAKE) -f CMakeFiles/jacobi.dir/build.make CMakeFiles/jacobi.dir/jacobi.c.o.provides.build
.PHONY : CMakeFiles/jacobi.dir/jacobi.c.o.provides

CMakeFiles/jacobi.dir/jacobi.c.o.provides.build: CMakeFiles/jacobi.dir/jacobi.c.o


# Object files for target jacobi
jacobi_OBJECTS = \
"CMakeFiles/jacobi.dir/jacobi.c.o"

# External object files for target jacobi
jacobi_EXTERNAL_OBJECTS =

jacobi: CMakeFiles/jacobi.dir/jacobi.c.o
jacobi: CMakeFiles/jacobi.dir/build.make
jacobi: CMakeFiles/jacobi.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/jason/school/cs569/programs/assignment2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable jacobi"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/jacobi.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/jacobi.dir/build: jacobi

.PHONY : CMakeFiles/jacobi.dir/build

CMakeFiles/jacobi.dir/requires: CMakeFiles/jacobi.dir/jacobi.c.o.requires

.PHONY : CMakeFiles/jacobi.dir/requires

CMakeFiles/jacobi.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/jacobi.dir/cmake_clean.cmake
.PHONY : CMakeFiles/jacobi.dir/clean

CMakeFiles/jacobi.dir/depend:
	cd /home/jason/school/cs569/programs/assignment2/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jason/school/cs569/programs/assignment2 /home/jason/school/cs569/programs/assignment2 /home/jason/school/cs569/programs/assignment2/build /home/jason/school/cs569/programs/assignment2/build /home/jason/school/cs569/programs/assignment2/build/CMakeFiles/jacobi.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/jacobi.dir/depend
