#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cerrno>

#include "printer.hpp"
#include "ascii.hpp"
#include "../common/global.hpp"
#include "../common/gpu.hpp"

#include "../intel/uarch.hpp"
#include "../intel/intel.hpp"
#include "../hsa/hsa.hpp"
#include "../hsa/uarch.hpp"
#include "../cuda/cuda.hpp"
#include "../cuda/uarch.hpp"

#ifdef _WIN32
  #define NOMINMAX
  #include <Windows.h>
#else
  #ifdef  __linux__
    #ifndef _POSIX_C_SOURCE
      #define _POSIX_C_SOURCE 199309L
    #endif
  #endif
  #include <sys/ioctl.h>
  #include <unistd.h>
#endif

#define max(a,b) (((a)>(b))?(a):(b))
#define MAX_ATTRIBUTES      100
#define MAX_TERM_SIZE       1024

typedef struct {
  int id;
  const char *name;
  const char *shortname;
} AttributeField;

// AttributeField IDs
//                         Used by
enum {
  ATTRIBUTE_NAME,          // ALL
  ATTRIBUTE_CHIP,          // ALL
  ATTRIBUTE_UARCH,         // ALL
  ATTRIBUTE_TECHNOLOGY,    // ALL
  ATTRIBUTE_GT,            // Intel
  ATTRIBUTE_FREQUENCY,     // ALL
  ATTRIBUTE_COMPUTE_UNITS, // HSA
  ATTRIBUTE_STREAMINGMP,   // CUDA
  ATTRIBUTE_CORESPERMP,    // CUDA
  ATTRIBUTE_CUDA_CORES,    // CUDA
  ATTRIBUTE_TENSOR_CORES,  // CUDA
  ATTRIBUTE_EUS,           // Intel
  ATTRIBUTE_L2,            // CUDA
  ATTRIBUTE_MEMORY,        // CUDA
  ATTRIBUTE_MEMORY_FREQ,   // CUDA
  ATTRIBUTE_BUS_WIDTH,     // CUDA
  ATTRIBUTE_PEAK,          // ALL
  ATTRIBUTE_PEAK_TENSOR    // CUDA
};

static const AttributeField ATTRIBUTE_INFO[] = {
  { ATTRIBUTE_NAME,          "Name:",                   "Name:" },
  { ATTRIBUTE_CHIP,          "GPU processor:",          "Processor:" },
  { ATTRIBUTE_UARCH,         "Microarchitecture:",      "uArch:" },
  { ATTRIBUTE_TECHNOLOGY,    "Technology:",             "Technology:" },
  { ATTRIBUTE_GT,            "Graphics Tier:",          "GT:" },
  { ATTRIBUTE_FREQUENCY,     "Max Frequency:",          "Max Freq.:" },
  { ATTRIBUTE_COMPUTE_UNITS, "Compute Units (CUs)",     "CUs" },
  { ATTRIBUTE_STREAMINGMP,   "SMs:",                    "SMs:" },
  { ATTRIBUTE_CORESPERMP,    "Cores/SM:",               "Cores/SM:" },
  { ATTRIBUTE_CUDA_CORES,    "CUDA Cores:",             "CUDA Cores:" },
  { ATTRIBUTE_TENSOR_CORES,  "Tensor Cores:",           "Tensor Cores:" },
  { ATTRIBUTE_EUS,           "Execution Units:",        "EUs:" },
  { ATTRIBUTE_L2,            "L2 Size:",                "L2 Size:" },
  { ATTRIBUTE_MEMORY,        "Memory:",                 "Memory:" },
  { ATTRIBUTE_MEMORY_FREQ,   "Memory frequency:",       "Memory freq.:" },
  { ATTRIBUTE_BUS_WIDTH,     "Bus width:",              "Bus width:" },
  { ATTRIBUTE_PEAK,          "Peak Performance:",       "Peak Perf.:" },
  { ATTRIBUTE_PEAK_TENSOR,   "Peak Performance (MMA):", "Peak Perf.(MMA):" },
};

struct terminal {
  int w;
  int h;
};

struct attribute {
  int type;
  char* value;
};

struct ascii {
  struct ascii_logo* art;
  char reset[100];
  struct attribute** attributes;
  uint32_t n_attributes_set;
  uint32_t additional_spaces;
  VENDOR vendor;
  STYLE style;
};

void setAttribute(struct ascii* art, int type, char* value) {
  art->attributes[art->n_attributes_set]->value = value;
  art->attributes[art->n_attributes_set]->type = type;
  art->n_attributes_set++;

  if(art->n_attributes_set > MAX_ATTRIBUTES) {
    printBug("Set %d attributes, while max value is %d!", art->n_attributes_set, MAX_ATTRIBUTES);
  }
}

char* rgb_to_ansi(struct color* c, bool background, bool bold) {
  char* str = (char *) emalloc(sizeof(char) * 100);
  if(background) {
    snprintf(str, 44, "\x1b[48;2;%.3d;%.3d;%.3dm", c->R, c->G, c->B);
  }
  else {
    if(bold)
      snprintf(str, 48, "\x1b[1m\x1b[38;2;%.3d;%.3d;%.3dm", c->R, c->G, c->B);
    else
      snprintf(str, 44, "\x1b[38;2;%.3d;%.3d;%.3dm", c->R, c->G, c->B);
  }

  return str;
}

struct ascii* set_ascii(VENDOR vendor, STYLE style) {
  struct ascii* art = (struct ascii*) emalloc(sizeof(struct ascii));

  art->n_attributes_set = 0;
  art->additional_spaces = 0;
  art->vendor = vendor;
  art->attributes = (struct attribute**) emalloc(sizeof(struct attribute *) * MAX_ATTRIBUTES);
  for(uint32_t i=0; i < MAX_ATTRIBUTES; i++) {
    art->attributes[i] = (struct attribute*) emalloc(sizeof(struct attribute));
    art->attributes[i]->type = 0;
    art->attributes[i]->value = NULL;
  }

  #ifdef _WIN32
    // Old Windows do not define the flag
    #ifndef ENABLE_VIRTUAL_TERMINAL_PROCESSING
      #define ENABLE_VIRTUAL_TERMINAL_PROCESSING 0x0004
    #endif

    HANDLE std_handle = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD console_mode;

    // Attempt to enable the VT100-processing flag
    GetConsoleMode(std_handle, &console_mode);
    SetConsoleMode(std_handle, console_mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
    // Get the console mode flag again, to see if it successfully enabled it
    GetConsoleMode(std_handle, &console_mode);
  #endif

  if(style == STYLE_EMPTY) {
    #ifdef _WIN32
      // Use fancy style if VT100-processing is enabled,
      // or legacy style in other case
      art->style = (console_mode & ENABLE_VIRTUAL_TERMINAL_PROCESSING) ? STYLE_FANCY : STYLE_LEGACY;
    #else
      art->style = STYLE_FANCY;
    #endif
  }
  else {
    art->style = style;
  }

  return art;
}

void parse_print_color(struct ascii* art, uint32_t* logo_pos) {
  struct ascii_logo* logo = art->art;
  char color_id_str = logo->art[*logo_pos + 2];

  if(color_id_str == 'R') {
    printf("%s", art->reset);
  }
  else {
    int color_id = (color_id_str - '0') - 1;
    printf("%s", logo->color_ascii[color_id]);
  }

  *logo_pos += 3;
}

bool ascii_fits_screen(int termw, struct ascii_logo logo, int lf) {
  return termw - ((int) logo.width + lf) >= 0;
}

// TODO: Instead of using a function to do so, change ascii.h
// and store an color ID that is converted to BG or FG depending
// on logo->replace_blocks
void replace_bgbyfg_color(struct ascii_logo* logo) {
  // Replace background by foreground color
  for(int i=0; i < 2; i++) {
    if(logo->color_ascii[i] == NULL) break;

    if(strcmp(logo->color_ascii[i], C_BG_BLACK) == 0) strcpy(logo->color_ascii[i], C_FG_BLACK);
    else if(strcmp(logo->color_ascii[i], C_BG_RED) == 0) strcpy(logo->color_ascii[i], C_FG_RED);
    else if(strcmp(logo->color_ascii[i], C_BG_GREEN) == 0) strcpy(logo->color_ascii[i], C_FG_GREEN);
    else if(strcmp(logo->color_ascii[i], C_BG_YELLOW) == 0) strcpy(logo->color_ascii[i], C_FG_YELLOW);
    else if(strcmp(logo->color_ascii[i], C_BG_BLUE) == 0) strcpy(logo->color_ascii[i], C_FG_BLUE);
    else if(strcmp(logo->color_ascii[i], C_BG_MAGENTA) == 0) strcpy(logo->color_ascii[i], C_FG_MAGENTA);
    else if(strcmp(logo->color_ascii[i], C_BG_CYAN) == 0) strcpy(logo->color_ascii[i], C_FG_CYAN);
    else if(strcmp(logo->color_ascii[i], C_BG_WHITE) == 0) strcpy(logo->color_ascii[i], C_FG_WHITE);
  }
}

struct ascii_logo* choose_ascii_art_aux(struct ascii_logo* logo_long, struct ascii_logo* logo_short, struct terminal* term, int lf) {
  if(show_logo_long()) return logo_long;
  if(show_logo_short()) return logo_short;
  if(ascii_fits_screen(term->w, *logo_long, lf)) {
    return logo_long;
  }
  else {
    return logo_short;
  }
}

void choose_ascii_art(struct ascii* art, struct color** cs, struct terminal* term, int lf) {
  if(art->vendor == GPU_VENDOR_NVIDIA) {
    art->art = choose_ascii_art_aux(&logo_nvidia_l, &logo_nvidia, term, lf);
  }
  else if(art->vendor == GPU_VENDOR_AMD) {
    art->art = choose_ascii_art_aux(&logo_amd_l, &logo_amd, term, lf);
  }
  else if(art->vendor == GPU_VENDOR_INTEL) {
    art->art = choose_ascii_art_aux(&logo_intel_l, &logo_intel, term, lf);
  }
  else {
    art->art = &logo_unknown;
  }

  // 2. Choose colors
  struct ascii_logo* logo = art->art;

  switch(art->style) {
    case STYLE_LEGACY:
      logo->replace_blocks = false;
      strcpy(logo->color_text[0], C_NONE);
      strcpy(logo->color_text[1], C_NONE);
      strcpy(logo->color_ascii[0], C_NONE);
      strcpy(logo->color_ascii[1], C_NONE);
      art->reset[0] = '\0';
      break;
    case STYLE_RETRO:
      logo->replace_blocks = false;
      replace_bgbyfg_color(logo);
      // fall through
    case STYLE_FANCY:
      if(cs != NULL) {
        strcpy(logo->color_text[0], rgb_to_ansi(cs[2], false, true));
        strcpy(logo->color_text[1], rgb_to_ansi(cs[3], false, true));
        strcpy(logo->color_ascii[0], rgb_to_ansi(cs[0], logo->replace_blocks, true));
        strcpy(logo->color_ascii[1], rgb_to_ansi(cs[1], logo->replace_blocks, true));
      }
      strcpy(art->reset, C_RESET);
      break;
    case STYLE_INVALID:
    default:
      printBug("Found invalid style (%d)", art->style);
  }
}

uint32_t longest_attribute_length(struct ascii* art, bool use_short) {
  uint32_t max = 0;
  uint64_t len = 0;

  for(uint32_t i=0; i < art->n_attributes_set; i++) {
    if(art->attributes[i]->value != NULL) {
      const char* str = use_short ? ATTRIBUTE_INFO[art->attributes[i]->type].shortname : ATTRIBUTE_INFO[art->attributes[i]->type].name;
      len = strlen(str);
      if(len > max) max = len;
    }
  }

  return max;
}

uint32_t longest_field_length(struct ascii* art, int la) {
  uint32_t max = 0;
  uint64_t len = 0;

  for(uint32_t i=0; i < art->n_attributes_set; i++) {
    if(art->attributes[i]->value != NULL) {
      // longest attribute + 1 (space) + longest value
      len = la + 1 + strlen(art->attributes[i]->value);

      if(len > max) max = len;
    }
  }

  return max;
}

void print_ascii_generic(struct ascii* art, uint32_t la, int32_t text_space, bool use_short) {
  struct ascii_logo* logo = art->art;
  int attr_to_print = 0;
  int attr_type;
  char* attr_value;
  int32_t current_space;
  int32_t space_right;
  int32_t space_up = ((int)logo->height - (int)art->n_attributes_set)/2;
  int32_t space_down = (int)logo->height - (int)art->n_attributes_set - (int)space_up;
  uint32_t logo_pos = 0;
  int32_t iters = max(logo->height, art->n_attributes_set);

  printf("\n");
  for(int32_t n=0; n < iters; n++) {
    // 1. Print logo
    if(space_up > 0 || (space_up + n >= 0 && space_up + n < (int)logo->height)) {
      for(uint32_t i=0; i < logo->width; i++) {
        if(logo->art[logo_pos] == '$') {
          if(logo->replace_blocks) logo_pos += 3;
          else parse_print_color(art, &logo_pos);
        }
        if(logo->replace_blocks && logo->art[logo_pos] != ' ') {
          if(logo->art[logo_pos] == '#') printf("%s%c%s", logo->color_ascii[0], ' ', art->reset);
          else if(logo->art[logo_pos] == '@') printf("%s%c%s", logo->color_ascii[1], ' ', art->reset);
          else printf("%c", logo->art[logo_pos]);
        }
        else
          printf("%c", logo->art[logo_pos]);

        logo_pos++;
      }
      printf("%s", art->reset);
    }
    else {
      // If logo should not be printed, fill with spaces
      printf("%*c", logo->width, ' ');
    }

    // 2. Print text
    if(space_up < 0 || (n > space_up-1 && n < (int)logo->height - space_down)) {
      attr_type = art->attributes[attr_to_print]->type;
      attr_value = art->attributes[attr_to_print]->value;
      attr_to_print++;

      const char* attr_str = use_short ? ATTRIBUTE_INFO[attr_type].shortname : ATTRIBUTE_INFO[attr_type].name;

      space_right = 1 + (la - strlen(attr_str));
      current_space = max(0, text_space);

      printf("%s%.*s%s", logo->color_text[0], current_space, attr_str, art->reset);
      current_space = max(0, current_space - (int) strlen(attr_str));
      printf("%*s", min(current_space, space_right), "");
      current_space = max(0, current_space - min(current_space, space_right));
      printf("%s%.*s%s", logo->color_text[1], current_space, attr_value, art->reset);
      printf("\n");
    }
    else printf("\n");
  }
  printf("\n");
}

#ifdef BACKEND_INTEL
bool print_gpufetch_intel(struct gpu_info* gpu, STYLE s, struct color** cs, struct terminal* term) {
  struct ascii* art = set_ascii(get_gpu_vendor(gpu), s);

  if(art == NULL)
    return false;

  char* gpu_name = get_str_gpu_name(gpu);
  char* uarch = get_str_uarch_intel(gpu->arch);
  char* gt = get_str_gt(gpu->arch);
  char* manufacturing_process = get_str_process(gpu->arch);
  char* eus = get_str_eu(gpu);
  char* max_frequency = get_str_freq(gpu);
  char* pp = get_str_peak_performance(gpu);

  setAttribute(art, ATTRIBUTE_NAME, gpu_name);
  setAttribute(art, ATTRIBUTE_UARCH, uarch);
  setAttribute(art, ATTRIBUTE_TECHNOLOGY, manufacturing_process);
  setAttribute(art, ATTRIBUTE_FREQUENCY, max_frequency);
  setAttribute(art, ATTRIBUTE_GT, gt);
  setAttribute(art, ATTRIBUTE_EUS, eus);
  setAttribute(art, ATTRIBUTE_PEAK, pp);

  bool use_short = false;
  uint32_t longest_attribute = longest_attribute_length(art, use_short);
  uint32_t longest_field = longest_field_length(art, longest_attribute);
  choose_ascii_art(art, cs, term, longest_field);

  if(!ascii_fits_screen(term->w, *art->art, longest_field)) {
    // Despite of choosing the smallest logo, the output does not fit
    // Choose the shorter field names and recalculate the longest attr
    use_short = true;
    longest_attribute = longest_attribute_length(art, use_short);
  }

  print_ascii_generic(art, longest_attribute, term->w - art->art->width, use_short);

  return true;
}
#endif

#ifdef BACKEND_CUDA
bool print_gpufetch_cuda(struct gpu_info* gpu, STYLE s, struct color** cs, struct terminal* term) {
  struct ascii* art = set_ascii(get_gpu_vendor(gpu), s);

  if(art == NULL)
    return false;

  char* gpu_name = get_str_gpu_name(gpu);
  char* gpu_chip = get_str_chip(gpu->arch);
  char* uarch = get_str_uarch_cuda(gpu->arch);
  char* comp_cap = get_str_cc(gpu->arch);
  char* manufacturing_process = get_str_process(gpu->arch);
  char* sms = get_str_sm(gpu);
  char* corespersm = get_str_cores_sm(gpu);
  char* cores = get_str_cuda_cores(gpu);
  char* tensorc = get_str_tensor_cores(gpu);
  char* max_frequency = get_str_freq(gpu);
  char* l2 = get_str_l2(gpu);
  char* mem_size = get_str_memory_size(gpu);
  char* mem_type = get_str_memory_type(gpu);
  char* mem_freq = get_str_memory_clock(gpu);
  char* bus_width = get_str_bus_width(gpu);
  char* pp = get_str_peak_performance(gpu);
  char* pp_tensor = get_str_peak_performance_tensor(gpu);

  char* mem = (char *) emalloc(sizeof(char) * (strlen(mem_size) + strlen(mem_type) + 2));
  sprintf(mem, "%s %s", mem_size, mem_type);

  char* uarch_cc = (char *) emalloc(sizeof(char) * (strlen(uarch) + strlen(comp_cap) + 4));
  sprintf(uarch_cc, "%s (%s)", uarch, comp_cap);

  setAttribute(art, ATTRIBUTE_NAME, gpu_name);
  setAttribute(art, ATTRIBUTE_CHIP, gpu_chip);
  setAttribute(art, ATTRIBUTE_UARCH, uarch_cc);
  setAttribute(art, ATTRIBUTE_TECHNOLOGY, manufacturing_process);
  setAttribute(art, ATTRIBUTE_FREQUENCY, max_frequency);
  setAttribute(art, ATTRIBUTE_STREAMINGMP, sms);
  setAttribute(art, ATTRIBUTE_CORESPERMP, corespersm);
  setAttribute(art, ATTRIBUTE_CUDA_CORES, cores);
  if(gpu->topo_c->tensor_cores > 0) {
    setAttribute(art, ATTRIBUTE_TENSOR_CORES, tensorc);
  }
  setAttribute(art, ATTRIBUTE_MEMORY, mem);
  setAttribute(art, ATTRIBUTE_MEMORY_FREQ, mem_freq);
  setAttribute(art, ATTRIBUTE_BUS_WIDTH, bus_width);
  setAttribute(art, ATTRIBUTE_L2, l2);
  setAttribute(art, ATTRIBUTE_PEAK, pp);
  if(gpu->topo_c->tensor_cores > 0) {
    setAttribute(art, ATTRIBUTE_PEAK_TENSOR, pp_tensor);
  }

  bool use_short = false;
  uint32_t longest_attribute = longest_attribute_length(art, use_short);
  uint32_t longest_field = longest_field_length(art, longest_attribute);
  choose_ascii_art(art, cs, term, longest_field);

  if(!ascii_fits_screen(term->w, *art->art, longest_field)) {
    // Despite of choosing the smallest logo, the output does not fit
    // Choose the shorter field names and recalculate the longest attr
    use_short = true;
    longest_attribute = longest_attribute_length(art, use_short);
  }

  print_ascii_generic(art, longest_attribute, term->w - art->art->width, use_short);

  free(manufacturing_process);
  free(max_frequency);
  free(l2);
  free(pp);

  free(art->attributes);
  free(art);

  return true;
}
#endif

#ifdef BACKEND_HSA
bool print_gpufetch_amd(struct gpu_info* gpu, STYLE s, struct color** cs, struct terminal* term) {
  struct ascii* art = set_ascii(get_gpu_vendor(gpu), s);

  if(art == NULL)
    return false;

  char* gpu_name = get_str_gpu_name(gpu);
  char* gpu_chip = get_str_chip(gpu->arch);
  char* uarch = get_str_uarch_hsa(gpu->arch);
  char* manufacturing_process = get_str_process(gpu->arch);
  char* cus = get_str_cu(gpu);
  char* max_frequency = get_str_freq(gpu);

  setAttribute(art, ATTRIBUTE_NAME, gpu_name);
  if (gpu_chip != NULL) {
    setAttribute(art, ATTRIBUTE_CHIP, gpu_chip);
  }
  setAttribute(art, ATTRIBUTE_UARCH, uarch);
  setAttribute(art, ATTRIBUTE_TECHNOLOGY, manufacturing_process);
  setAttribute(art, ATTRIBUTE_FREQUENCY, max_frequency);
  setAttribute(art, ATTRIBUTE_COMPUTE_UNITS, cus);

  bool use_short = false;
  uint32_t longest_attribute = longest_attribute_length(art, use_short);
  uint32_t longest_field = longest_field_length(art, longest_attribute);
  choose_ascii_art(art, cs, term, longest_field);

  if(!ascii_fits_screen(term->w, *art->art, longest_field)) {
    // Despite of choosing the smallest logo, the output does not fit
    // Choose the shorter field names and recalculate the longest attr
    use_short = true;
    longest_attribute = longest_attribute_length(art, use_short);
  }

  print_ascii_generic(art, longest_attribute, term->w - art->art->width, use_short);

  free(art->attributes);
  free(art);

  return true;
}
#endif

struct terminal* get_terminal_size() {
  struct terminal* term = (struct terminal*) emalloc(sizeof(struct terminal));

#ifdef _WIN32
  CONSOLE_SCREEN_BUFFER_INFO csbi;
  if(GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi) == 0) {
    printWarn("GetConsoleScreenBufferInfo failed");
    term->w = MAX_TERM_SIZE;
    term->h = MAX_TERM_SIZE;
    return term;
  }
  term->w = csbi.srWindow.Right - csbi.srWindow.Left + 1;
  term->h = csbi.srWindow.Bottom - csbi.srWindow.Top + 1;
#else
  struct winsize w;
  if(ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == -1) {
    printWarn("get_terminal_size: ioctl: %s", strerror(errno));
    term->w = MAX_TERM_SIZE;
    term->h = MAX_TERM_SIZE;
    return term;
  }
  term->h = w.ws_row;
  term->w = w.ws_col;
#endif

  return term;
}

bool print_gpufetch(struct gpu_info* gpu, STYLE s, struct color** cs) {
  struct terminal* term = get_terminal_size();

  if(gpu->vendor == GPU_VENDOR_NVIDIA) {
    #ifdef BACKEND_CUDA
      if(clean_output()) printf("%*s", (int) strlen(CUDA_DRIVER_START_WARNING), " ");
      return print_gpufetch_cuda(gpu, s, cs, term);
    #else
      return false;
    #endif
  }
  else if(gpu->vendor == GPU_VENDOR_AMD) {
    #ifdef BACKEND_HSA
      return print_gpufetch_amd(gpu, s, cs, term);
    #else
      return false;
    #endif
  }
  else if(gpu->vendor == GPU_VENDOR_INTEL) {
    #ifdef BACKEND_INTEL
      return print_gpufetch_intel(gpu, s, cs, term);
    #else
      return false;
    #endif
  }
  else {
    printErr("Invalid GPU vendor: %d", gpu->vendor);
    return false;
  }
}
