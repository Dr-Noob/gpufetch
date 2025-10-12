#ifndef __ASCII__
#define __ASCII__

#include "colors.hpp"

struct ascii_logo {
  const char* art;
  uint32_t width;
  uint32_t height;
  bool replace_blocks;
  char color_ascii[8][100];
  char color_text[2][100];
};

// SHORT LOGOS //
#define ASCII_NVIDIA \
"$C1               'cccccccccccccccccccccccccc   \
$C1               ;oooooooooooooooooooooooool   \
$C1           .:::.     .oooooooooooooooooool   \
$C1      .:cll;   ,c:::.     cooooooooooooool   \
$C1   ,clo'      ;.   oolc:     ooooooooooool   \
$C1.cloo    ;cclo .      .olc.    coooooooool   \
$C1oooo   :lo,    ;ll;    looc    :oooooooool   \
$C1 oooc   ool.   ;oooc;clol    :looooooooool   \
$C1  :ooc   ,ol;  ;oooooo.   .cloo;     loool   \
$C1    ool;   .olc.       ,:lool        .lool   \
$C1      ool:.    ,::::ccloo.        :clooool   \
$C1         oolc::.            ':cclooooooool   \
$C1               ;oooooooooooooooooooooooool   \
$C1                                             \
$C1                                             \
$C2######.  ##   ##  ##  ######   ##    ###     \
$C2##   ##  ##   ##  ##  ##   ##  ##   #: :#    \
$C2##   ##   ## ##   ##  ##   ##  ##  #######   \
$C2##   ##    ###    ##  ######   ## ##     ##  "

#define ASCII_AMD \
"$C2          '###############             \
$C2             ,#############            \
$C2                      .####            \
$C2              #.      .####            \
$C2            :##.      .####            \
$C2           :###.      .####            \
$C2           #########.   :##            \
$C2           #######.       ;            \
$C1                                       \
$C1    ###     ###      ###   #######     \
$C1   ## ##    #####  #####   ##     ##   \
$C1  ##   ##   ### #### ###   ##      ##  \
$C1 #########  ###  ##  ###   ##      ##  \
$C1##       ## ###      ###   ##     ##   \
$C1##       ## ###      ###   #######     "

#define ASCII_INTEL \
"$C1                   .#################.          \
$C1              .####                   ####.     \
$C1          .##                             ###   \
$C1       ##                          :##     ###  \
$C1    #                ##            :##      ##  \
$C1  ##   ##  ######.   ####  ######  :##      ##  \
$C1 ##    ##  ##:  ##:  ##   ##   ### :##     ###  \
$C1##     ##  ##:  ##:  ##  :######## :##    ##    \
$C1##     ##  ##:  ##:  ##   ##.   .  :## ####     \
$C1##      #  ##:  ##:  ####  #####:   ##          \
$C1 ##                                             \
$C1  ###.                         ..o####.         \
$C1   ######oo...         ..oo#######              \
$C1          o###############o                     "

// LONG LOGOS
#define ASCII_NVIDIA_L \
"$C1                  MMMMMMMMMMMMMMMMMMMMMMMMMMMMMM  \
$C1                  MMMMMMMMMMMMMMMMMMMMMMMMMMMMMM  \
$C1                .::   'MMMMMMMMMMMMMMMMMMMMMMMMM  \
$C1           ccllooo;:;.       ;MMMMMMMMMMMMMMMMMM  \
$C1       cloc       :ooollcc:     :MMMMMMMMMMMMMMM  \
$C1    cloc      :ccl;      lolc,     ;MMMMMMMMMMMM  \
$C1.cloo:    :clo    ;c:      .ool;     MMMMMMMMMMM  \
$C1  ooo:    ooo     :ool,  .cloo.    ;lMMMMMMMMMMM  \
$C1   ooo:    ooc    :ooooccooo.    :MMMM  lMMMMMMM  \
$C1     ooc.   ool:  :oooooo'    ,cloo.        MMMM  \
$C1      ool:.    olc:       .:cloo.          :MMMM  \
$C1         olc,     ;:::cccloo.          :MMMMMMMM  \
$C1            olcc::;              ,:ccloMMMMMMMMM  \
$C1                  :......oMMMMMMMMMMMMMMMMMMMMMM  \
$C1                  :lllMMMMMMMMMMMMMMMMMMMMMMMMMM  "

#define ASCII_AMD_L \
"$C1                                                              \
$C1                                                              \
$C1                                                              \
$C1                                                              \
$C1                                                              \
$C1                                                              \
$C1     @@@@      @@@       @@@   @@@@@@@@      $C2  ############   \
$C1    @@@@@@     @@@@@   @@@@@   @@@    @@@    $C2    ##########   \
$C1   @@@  @@@    @@@@@@@@@@@@@   @@@      @@   $C2   #     #####   \
$C1  @@@    @@@   @@@  @@@  @@@   @@@      @@   $C2 ###     #####   \
$C1 @@@@@@@@@@@@  @@@       @@@   @@@    @@@    $C2#########  ###   \
$C1 @@@      @@@  @@@       @@@   @@@@@@@@@     $C2########    ##   \
$C1                                                              \
$C1                                                              \
$C1                                                              \
$C1                                                              \
$C1                                                              \
$C1                                                              \
$C1                                                              "

#define ASCII_INTEL_L \
"$C1                               ###############@               \
$C1                       ######@                ######@         \
$C1                  ###@                              ###@      \
$C1              ##@                                     ###@    \
$C1         ##@                                             ##@  \
$C1         ##@                                             ##@  \
$C1      @                    ##@                ##@        ##@  \
$C1    #@   ##@   ########@   #####@   #####@    ##@        ##@  \
$C1   #@    ##@   ##@    ##@  ##@    ###@  ###@  ##@        ##@  \
$C1  #@     ##@   ##@    ##@  ##@    ##@    ##@  ##@       ##@   \
$C1 #@      ##@   ##@    ##@  ##@    #########@  ##@     ###@    \
$C1 #@      ##@   ##@    ##@  ##@    ##@         ##@   ####@     \
$C1 #@       #@   ##@    ##@   ####@  ########@   #@  ##@        \
$C1 ##@                                                          \
$C1  ##@                                                         \
$C1  ###@                                        ###@            \
$C1    ####@                               #########@            \
$C1      #########@               ###############@               \
$C1          ##############################@                     "

typedef struct ascii_logo asciiL;

//                      ------------------------------------------------------------------------------------------
//                      | LOGO            | W | H | REPLACE | COLORS LOGO           | COLORS TEXT                |
//                      ------------------------------------------------------------------------------------------
asciiL logo_nvidia    = { ASCII_NVIDIA,    45, 19, false, {C_FG_GREEN, C_FG_WHITE}, {C_FG_WHITE, C_FG_GREEN}   };
asciiL logo_amd       = { ASCII_AMD,       39, 15, false, {C_FG_WHITE, C_FG_GREEN}, {C_FG_WHITE, C_FG_GREEN}   };
asciiL logo_intel     = { ASCII_INTEL,     48, 14, false, {C_FG_CYAN},              {C_FG_CYAN,  C_FG_WHITE}   };
// Long variants        | ---------------------------------------------------------------------------------------|
asciiL logo_nvidia_l  = { ASCII_NVIDIA_L,  50, 15, false, {C_FG_GREEN, C_FG_WHITE}, {C_FG_WHITE, C_FG_GREEN}   };
asciiL logo_amd_l     = { ASCII_AMD_L,     62, 19, true,  {C_BG_WHITE, C_BG_WHITE}, {C_FG_CYAN,  C_FG_B_WHITE} };
asciiL logo_intel_l   = { ASCII_INTEL_L,   62, 19, true,  {C_BG_CYAN, C_BG_WHITE},  {C_FG_CYAN,  C_FG_WHITE}   };
asciiL logo_unknown   = { NULL,            0,  0,  false, {C_NONE},                 {C_NONE,     C_NONE}       };

#endif
