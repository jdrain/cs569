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
CMAKE_SOURCE_DIR = /home/jason/school/cs569/programs/assignment1/Assignment_1

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jason/school/cs569/programs/assignment1/Assignment_1/build

# Include any dependencies generated for this target.
include CMakeFiles/mm.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/mm.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/mm.dir/flags.make

CMakeFiles/mm.dir/mm.c.o: CMakeFiles/mm.dir/flags.make
CMakeFiles/mm.dir/mm.c.o: ../mm.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jason/school/cs569/programs/assignment1/Assignment_1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/mm.dir/mm.c.o"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/mm.dir/mm.c.o   -c /home/jason/school/cs569/programs/assignment1/Assignment_1/mm.c

CMakeFiles/mm.dir/mm.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/mm.dir/mm.c.i"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/jason/school/cs569/programs/assignment1/Assignment_1/mm.c > CMakeFiles/mm.dir/mm.c.i

CMakeFiles/mm.dir/mm.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/mm.dir/mm.c.s"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/jason/school/cs569/programs/assignment1/Assignment_1/mm.c -o CMakeFiles/mm.dir/mm.c.s

CMakeFiles/mm.dir/mm.c.o.requires:

.PHONY : CMakeFiles/mm.dir/mm.c.o.requires

CMakeFiles/mm.dir/mm.c.o.provides: CMakeFiles/mm.dir/mm.c.o.requires
	$(MAKE) -f CMakeFiles/mm.dir/build.make CMakeFiles/mm.dir/mm.c.o.provides.build
.PHONY : CMakeFiles/mm.dir/mm.c.o.provides

CMakeFiles/mm.dir/mm.c.o.provides.build: CMakeFiles/mm.dir/mm.c.o


# Object files for target mm
mm_OBJECTS = \
"CMakeFiles/mm.dir/mm.c.o"

# External object files for target mm
mm_EXTERNAL_OBJECTS =

mm: CMakeFiles/mm.dir/mm.c.o
mm: CMakeFiles/mm.dir/build.make
mm: CMakeFiles/mm.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/jason/school/cs569/programs/assignment1/Assignment_1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable mm"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/mm.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/mm.dir/build: mm

.PHONY : CMakeFiles/mm.dir/build

CMakeFiles/mm.dir/requires: CMakeFiles/mm.dir/mm.c.o.requires

.PHONY : CMakeFiles/mm.dir/requires

CMakeFiles/mm.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/mm.dir/cmake_clean.cmake
.PHONY : CMakeFiles/mm.dir/clean

CMakeFiles/mm.dir/depend:
	cd /home/jason/school/cs569/programs/assignment1/Assignment_1/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jason/school/cs569/programs/assignment1/Assignment_1 /home/jason/school/cs569/programs/assignment1/Assignment_1 /home/jason/school/cs569/programs/assignment1/Assignment_1/build /home/jason/school/cs569/programs/assignment1/Assignment_1/build /home/jason/school/cs569/programs/assignment1/Assignment_1/build/CMakeFiles/mm.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/mm.dir/depend

