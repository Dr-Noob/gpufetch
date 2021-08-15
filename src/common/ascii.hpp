#ifndef __ASCII__
#define __ASCII__

#define COLOR_NONE         ""
#define COLOR_FG_BLACK     "\x1b[30;1m"
#define COLOR_FG_RED       "\x1b[31;1m"
#define COLOR_FG_GREEN     "\x1b[32;1m"
#define COLOR_FG_YELLOW    "\x1b[33;1m"
#define COLOR_FG_BLUE      "\x1b[34;1m"
#define COLOR_FG_MAGENTA   "\x1b[35;1m"
#define COLOR_FG_CYAN      "\x1b[36;1m"
#define COLOR_FG_WHITE     "\x1b[37;1m"
#define COLOR_BG_BLACK     "\x1b[40;1m"
#define COLOR_BG_RED       "\x1b[41;1m"
#define COLOR_BG_GREEN     "\x1b[42;1m"
#define COLOR_BG_YELLOW    "\x1b[43;1m"
#define COLOR_BG_BLUE      "\x1b[44;1m"
#define COLOR_BG_MAGENTA   "\x1b[45;1m"
#define COLOR_BG_CYAN      "\x1b[46;1m"
#define COLOR_BG_WHITE     "\x1b[47;1m"
#define COLOR_FG_B_BLACK   "\x1b[90;1m"
#define COLOR_FG_B_RED     "\x1b[91;1m"
#define COLOR_FG_B_GREEN   "\x1b[92;1m"
#define COLOR_FG_B_YELLOW  "\x1b[93;1m"
#define COLOR_FG_B_BLUE    "\x1b[94;1m"
#define COLOR_FG_B_MAGENTA "\x1b[95;1m"
#define COLOR_FG_B_CYAN    "\x1b[96;1m"
#define COLOR_FG_B_WHITE   "\x1b[97;1m"
#define COLOR_RESET        "\x1b[m"

struct ascii_logo {
  char* art;
  uint32_t width;
  uint32_t height;
  bool replace_blocks;
  char color_ascii[8][100];
  char color_text[2][100];
};

// SHORT LOGOS //
#define ASCII_NVIDIA \
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

// LONG LOGOS //
#define ASCII_NVIDIA_L \
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

//                      -----------------------------------------------------------------------------------------------------
//                      | LOGO          | W | H | REPLACE | COLORS LOGO (>0 && <10)        | COLORS TEXT (=2)               |
//                      -----------------------------------------------------------------------------------------------------
asciiL logo_nvidia    = { ASCII_NVIDIA,    48, 14, false, {COLOR_FG_CYAN},                 {COLOR_FG_CYAN, COLOR_FG_WHITE} };
// Long variants        | --------------------------------------------------------------------------------------------------|
asciiL logo_nvidia_l  = { ASCII_NVIDIA_L,  62, 19, true,  {COLOR_BG_CYAN, COLOR_BG_WHITE}, {COLOR_FG_CYAN, COLOR_FG_WHITE} };
asciiL logo_unknown   = { NULL,            0,  0, false,  {COLOR_NONE},                    {COLOR_NONE,    COLOR_NONE}     };

#endif
