#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>
#include <errno.h>

#include "printer.hpp"
#include "ascii.hpp"
#include "../common/global.hpp"
#include "../common/gpu.hpp"

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

enum {
  ATTRIBUTE_NAME,
  ATTRIBUTE_CHIP,
  ATTRIBUTE_UARCH,
  ATTRIBUTE_TECHNOLOGY,
  ATTRIBUTE_FREQUENCY,
  ATTRIBUTE_STREAMINGMP,
  ATTRIBUTE_CORESPERMP,
  ATTRIBUTE_CUDA_CORES,
  ATTRIBUTE_L2,
  ATTRIBUTE_MEMORY,
  ATTRIBUTE_MEMORY_FREQ,
  ATTRIBUTE_BUS_WIDTH,
  ATTRIBUTE_PEAK
};

static const char* ATTRIBUTE_FIELDS [] = {
  "Name:",
  "GPU processor:",
  "Microarchitecture:",
  "Technology:",
  "Max Frequency:",
  "SMs:",
  "Cores/SM:",
  "CUDA cores:",
  "L2 Size:",
  "Memory:",
  "Memory frequency:",
  "Bus width:",
  "Peak Performance:",
};

static const char* ATTRIBUTE_FIELDS_SHORT [] = {
  "Name:",
  "Processor:",
  "uArch:",
  "Technology:",
  "Max Freq.:",
  "SMs:",
  "Cores/SM:",
  "CUDA cores:",
  "L2 Size:",
  "Memory:",
  "Memory freq.:",
  "Bus width:",
  "Peak Perf.:",
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

    if(strcmp(logo->color_ascii[i], COLOR_BG_BLACK) == 0) strcpy(logo->color_ascii[i], COLOR_FG_BLACK);
    else if(strcmp(logo->color_ascii[i], COLOR_BG_RED) == 0) strcpy(logo->color_ascii[i], COLOR_FG_RED);
    else if(strcmp(logo->color_ascii[i], COLOR_BG_GREEN) == 0) strcpy(logo->color_ascii[i], COLOR_FG_GREEN);
    else if(strcmp(logo->color_ascii[i], COLOR_BG_YELLOW) == 0) strcpy(logo->color_ascii[i], COLOR_FG_YELLOW);
    else if(strcmp(logo->color_ascii[i], COLOR_BG_BLUE) == 0) strcpy(logo->color_ascii[i], COLOR_FG_BLUE);
    else if(strcmp(logo->color_ascii[i], COLOR_BG_MAGENTA) == 0) strcpy(logo->color_ascii[i], COLOR_FG_MAGENTA);
    else if(strcmp(logo->color_ascii[i], COLOR_BG_CYAN) == 0) strcpy(logo->color_ascii[i], COLOR_FG_CYAN);
    else if(strcmp(logo->color_ascii[i], COLOR_BG_WHITE) == 0) strcpy(logo->color_ascii[i], COLOR_FG_WHITE);
  }
}

void choose_ascii_art(struct ascii* art, struct color** cs, struct terminal* term, int lf) {
  if(art->vendor == GPU_VENDOR_NVIDIA) {
    if(term != NULL && ascii_fits_screen(term->w, logo_nvidia_l, lf))
      art->art = &logo_nvidia_l;
    else
      art->art = &logo_nvidia;
  }
  else {
    art->art = &logo_unknown;
  }

  // 2. Choose colors
  struct ascii_logo* logo = art->art;

  switch(art->style) {
    case STYLE_LEGACY:
      logo->replace_blocks = false;
      strcpy(logo->color_text[0], COLOR_NONE);
      strcpy(logo->color_text[1], COLOR_NONE);
      strcpy(logo->color_ascii[0], COLOR_NONE);
      strcpy(logo->color_ascii[1], COLOR_NONE);
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
      strcpy(art->reset, COLOR_RESET);
      break;
    case STYLE_INVALID:
    default:
      printBug("Found invalid style (%d)", art->style);
  }
}

uint32_t longest_attribute_length(struct ascii* art, const char** attribute_fields) {
  uint32_t max = 0;
  uint64_t len = 0;

  for(uint32_t i=0; i < art->n_attributes_set; i++) {
    if(art->attributes[i]->value != NULL) {
      len = strlen(attribute_fields[art->attributes[i]->type]);
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

void print_ascii_generic(struct ascii* art, uint32_t la, int32_t text_space, const char** attribute_fields) {
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

      space_right = 1 + (la - strlen(attribute_fields[attr_type]));
      current_space = max(0, text_space);

      printf("%s%.*s%s", logo->color_text[0], current_space, attribute_fields[attr_type], art->reset);
      current_space = max(0, current_space - (int) strlen(attribute_fields[attr_type]));
      printf("%*s", min(current_space, space_right), "");
      current_space = max(0, current_space - min(current_space, space_right));
      printf("%s%.*s%s", logo->color_text[1], current_space, attr_value, art->reset);
      printf("\n");
    }
    else printf("\n");
  }
  printf("\n");
}

bool print_gpufetch_cuda(struct gpu_info* gpu, STYLE s, struct color** cs, struct terminal* term) {
  struct ascii* art = set_ascii(get_gpu_vendor(gpu), s);

  if(art == NULL)
    return false;

  char* gpu_name = get_str_gpu_name(gpu);
  char* gpu_chip = get_str_chip(gpu->arch);
  char* uarch = get_str_uarch(gpu->arch);
  char* comp_cap = get_str_cc(gpu->arch);
  char* manufacturing_process = get_str_process(gpu->arch);
  char* sms = get_str_sm(gpu);
  char* corespersm = get_str_cores_sm(gpu);
  char* cores = get_str_cuda_cores(gpu);
  char* max_frequency = get_str_freq(gpu);
  char* l2 = get_str_l2(gpu);
  char* mem_size = get_str_memory_size(gpu);
  char* mem_type = get_str_memory_type(gpu);
  char* mem_freq = get_str_memory_clock(gpu);
  char* bus_width = get_str_bus_width(gpu);
  char* pp = get_str_peak_performance(gpu);

  char* mem = (char *) emalloc(sizeof(char) * (strlen(mem_size) + strlen(mem_type) + 2));
  sprintf(mem, "%s %s", mem_size, mem_type);

  char* uarch_cc = (char *) emalloc(sizeof(char) * (strlen(uarch) + strlen(comp_cap) + 3));
  sprintf(uarch_cc, "%s (%s)", uarch, comp_cap);

  setAttribute(art, ATTRIBUTE_NAME, gpu_name);
  setAttribute(art, ATTRIBUTE_CHIP, gpu_chip);
  setAttribute(art, ATTRIBUTE_UARCH, uarch_cc);
  setAttribute(art, ATTRIBUTE_TECHNOLOGY, manufacturing_process);
  setAttribute(art, ATTRIBUTE_FREQUENCY, max_frequency);
  setAttribute(art, ATTRIBUTE_STREAMINGMP, sms);
  setAttribute(art, ATTRIBUTE_CORESPERMP, corespersm);
  setAttribute(art, ATTRIBUTE_CUDA_CORES, cores);
  setAttribute(art, ATTRIBUTE_MEMORY, mem);
  setAttribute(art, ATTRIBUTE_MEMORY_FREQ, mem_freq);
  setAttribute(art, ATTRIBUTE_BUS_WIDTH, bus_width);
  setAttribute(art, ATTRIBUTE_L2, l2);
  setAttribute(art, ATTRIBUTE_PEAK, pp);

  const char** attribute_fields = ATTRIBUTE_FIELDS;
  uint32_t longest_attribute = longest_attribute_length(art, attribute_fields);
  uint32_t longest_field = longest_field_length(art, longest_attribute);
  choose_ascii_art(art, cs, term, longest_field);

  if(!ascii_fits_screen(term->w, *art->art, longest_field)) {
    // Despite of choosing the smallest logo, the output does not fit
    // Choose the shorter field names and recalculate the longest attr
    attribute_fields = ATTRIBUTE_FIELDS_SHORT;
    longest_attribute = longest_attribute_length(art, attribute_fields);
  }

  print_ascii_generic(art, longest_attribute, term->w - art->art->width, attribute_fields);

  free(manufacturing_process);
  free(max_frequency);
  free(l2);
  free(pp);

  free(art->attributes);
  free(art);

  /* if(cs != NULL) free_colors_struct(cs);
  free_cache_struct(cpu->cach);
  free_topo_struct(cpu->topo);
  free_freq_struct(cpu->freq);
  free_cpuinfo_struct(cpu);*/

  return true;
}

struct terminal* get_terminal_size() {
  struct terminal* term = (struct terminal*) emalloc(sizeof(struct terminal));

#ifdef _WIN32
  CONSOLE_SCREEN_BUFFER_INFO csbi;
  if(GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi) == 0) {
    printWarn("GetConsoleScreenBufferInfo failed");
    return NULL;
  }
  term->w = csbi.srWindow.Right - csbi.srWindow.Left + 1;
  term->h = csbi.srWindow.Bottom - csbi.srWindow.Top + 1;
#else
  struct winsize w;
  if(ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == -1) {
    printErr("ioctl: %s", strerror(errno));
    return NULL;
  }
  term->h = w.ws_row;
  term->w = w.ws_col;
#endif

  return term;
}

bool print_gpufetch(struct gpu_info* gpu, STYLE s, struct color** cs) {
  struct terminal* term = get_terminal_size();

  return print_gpufetch_cuda(gpu, s, cs, term);
}
