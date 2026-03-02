# Legend of Elya - N64 Homebrew ROM
# World's First LLM-powered Nintendo 64 Game
#
# Two ROMs:
#   make base    → legend_of_elya.z64         (LLM demo)
#   make mining  → legend_of_elya_mining.z64  (LLM + RustChain mining)
#   make all     → both

N64_INST ?= /home/sophia5070node/n64dev/mips64-toolchain
BUILD_DIR = build
BUILD_DIR_MINING = build_mining
include $(N64_INST)/n64.mk

# ─── Base ROM (LLM demo only) ────────────────────────────────────────────────
all: legend_of_elya.z64 legend_of_elya_mining.z64

base: legend_of_elya.z64

$(BUILD_DIR)/legend_of_elya.dfs: filesystem/sophia_weights.bin

$(BUILD_DIR)/legend_of_elya.elf: $(BUILD_DIR)/legend_of_elya.o $(BUILD_DIR)/nano_gpt.o

legend_of_elya.z64: N64_ROM_TITLE="Legend of Elya"
legend_of_elya.z64: $(BUILD_DIR)/legend_of_elya.dfs

# ─── Mining ROM (LLM + RustChain attestation) ────────────────────────────────
mining: legend_of_elya_mining.z64

$(BUILD_DIR_MINING)/legend_of_elya_mining.dfs: filesystem/sophia_weights.bin
	@mkdir -p $(BUILD_DIR_MINING)
	$(N64_MKDFS) $@ filesystem/

$(BUILD_DIR_MINING)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(N64_CFLAGS) -Imining/n64 -c $< -o $@

$(BUILD_DIR_MINING)/mining/n64/%.o: mining/n64/%.c
	@mkdir -p $(dir $@)
	$(CC) $(N64_CFLAGS) -Imining/n64 -c $< -o $@

$(BUILD_DIR_MINING)/legend_of_elya_mining.elf: $(BUILD_DIR_MINING)/legend_of_elya_mining.o $(BUILD_DIR_MINING)/nano_gpt.o $(BUILD_DIR_MINING)/mining/n64/n64_attest.o
	$(LD) $(N64_LDFLAGS) -o $@ $^ $(N64_LIBS)

legend_of_elya_mining.z64: N64_ROM_TITLE="Elya Mining"
legend_of_elya_mining.z64: $(BUILD_DIR_MINING)/legend_of_elya_mining.elf $(BUILD_DIR_MINING)/legend_of_elya_mining.dfs
	@echo "    [Z64] $@"
	$(N64_ROM) $@ $< -s 8M
	$(N64_ROMDFS) $@ $(BUILD_DIR_MINING)/legend_of_elya_mining.dfs

clean:
	rm -rf $(BUILD_DIR) $(BUILD_DIR_MINING) legend_of_elya.z64 legend_of_elya_mining.z64

-include $(wildcard $(BUILD_DIR)/*.d)
-include $(wildcard $(BUILD_DIR_MINING)/*.d)
-include $(wildcard $(BUILD_DIR_MINING)/mining/n64/*.d)

.PHONY: all base mining clean
