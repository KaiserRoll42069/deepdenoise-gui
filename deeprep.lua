--[[
  Deep-Rep Darktable Integration Plugin
  Interfaces Darktable with the Deep-Rep multi-frame neural restoration toolkit.
  Supports both interactive forwarding to deeprep_gui.py and direct background batch processing
  with automatic import of Adobe Linear RAW DNGs into Darktable.

  Original Deep-Rep Framework Copyright (c) 2021 Huawei Technologies Co., Ltd.
  Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International)
  https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
  Academic / non-commercial research use only.
--]]

local dt = require "darktable"
local du = require "lib/dtutils"
local df = require "lib/dtutils.file"
local dsys = require "lib/dtutils.system"

du.check_min_api_version("7.0.0", "deeprep")

local gettext = dt.gettext.gettext
local function _(msgid)
    return gettext(msgid)
end

local script_data = {}
script_data.metadata = {
    name = _("Deep-Rep Burst Restoration"),
    purpose = _("Restore RAW bursts using Deep-Rep neural super-resolution and denoising"),
    author = "Deep-Rep Toolkit",
    help = "https://github.com/goutamgmb/deep-rep"
}

script_data.destroy = nil
script_data.destroy_method = nil
script_data.restart = nil
script_data.show = nil

local MOD = "deeprep"
local is_windows = (dt.configuration.running_os == "windows")

-- Default configuration paths
local default_python = "python3"
local default_script = "deeprep_gui.py"

if is_windows then
    local local_app = os.getenv("LOCALAPPDATA") or "C:\\Users\\Admin\\AppData\\Local"
    local py312 = local_app .. "\\Programs\\Python\\Python312\\python.exe"
    local py311 = local_app .. "\\Programs\\Python\\Python311\\python.exe"
    local user_prof = os.getenv("USERPROFILE") or "C:\\Users\\Admin"
    local scratch_script = user_prof .. "\\.gemini\\antigravity-ide\\scratch\\deep-rep\\deeprep_gui.py"

    if df.check_if_file_exists(py312) then
        default_python = py312
    elseif df.check_if_file_exists(py311) then
        default_python = py311
    else
        default_python = "python"
    end

    if df.check_if_file_exists(scratch_script) then
        default_script = scratch_script
    else
        default_script = "deeprep_gui.py"
    end
else
    default_python = "python3"
    default_script = "/home/aronwsl/deep-rep/deeprep_gui.py"
end

-- Register Darktable Preferences
dt.preferences.register(MOD, "python_bin", "string", _("Python Executable / Command"), _("Path to Python executable or WSL command"), default_python)
dt.preferences.register(MOD, "script_path", "string", _("DeepRep GUI Script Path"), _("Absolute path to deeprep_gui.py"), default_script)
dt.preferences.register(MOD, "auto_import", "bool", _("Auto-Import Restored DNG"), _("Automatically import restored DNG back into darktable library"), true)

-- Helper Functions
local function clean_string(str)
    if not str then return "" end
    str = string.gsub(str, '^%s*', '')
    return string.gsub(str, '%s*$', '')
end

local function get_image_full_path(img)
    local sep = is_windows and "\\" or "/"
    if string.sub(img.path, -1) == "/" or string.sub(img.path, -1) == "\\" then
        return img.path .. img.filename
    else
        return img.path .. sep .. img.filename
    end
end

-- Determine Keyframe and Extra Frames based on selection and user criteria
local function separate_keyframe_and_extras(images, method_index, explicit_keyframe_name)
    if #images == 0 then return nil, {} end
    if #images == 1 then return images[1], {} end

    local keyframe = nil
    local extras = {}

    if method_index == 1 then
        -- 1. Explicit selection from dropdown
        if explicit_keyframe_name and explicit_keyframe_name ~= "" then
            for _, img in ipairs(images) do
                if img.filename == explicit_keyframe_name then
                    keyframe = img
                    break
                end
            end
        end
    elseif method_index == 2 then
        -- 2. Highest Star Rating
        local highest_rating = -1
        for _, img in ipairs(images) do
            if img.rating > highest_rating then
                highest_rating = img.rating
                keyframe = img
            end
        end
    elseif method_index == 3 then
        -- 3. Red Color Label
        for _, img in ipairs(images) do
            if img.red then
                keyframe = img
                break
            end
        end
    end

    -- Default fallback: First selected image
    if not keyframe then
        keyframe = images[1]
    end

    for _, img in ipairs(images) do
        if img.id ~= keyframe.id then
            table.insert(extras, img)
        end
    end

    return keyframe, extras
end

-- Convert Windows path to WSL path if running Python via WSL
local function normalize_path_for_command(path, python_cmd)
    if is_windows and string.find(python_cmd, "wsl") then
        local drive, rest = string.match(path, "^([A-Za-z]):[\\/](.*)")
        if drive and rest then
            rest = string.gsub(rest, "\\", "/")
            return "/mnt/" .. string.lower(drive) .. "/" .. rest
        end
    end
    -- Standard quote wrapping
    return path
end

-- Main UI Elements Table
local GUI = {
    keyframe_method = {},
    keyframe_selector = {},
    module = {},
    sr_scale = {},
    tile_size = {},
    use_fp16 = {},
    auto_import = {},
    status_label = {},
    btn_open_gui = {},
    btn_run_batch = {}
}

-- Build and Execute Command
local function execute_deeprep(headless)
    local selected_images = dt.gui.selection()
    if #selected_images < 1 then
        dt.print(_("Deep-Rep: Please select at least 1 RAW image in lighttable."))
        return
    end

    local python_bin = clean_string(dt.preferences.read(MOD, "python_bin", "string"))
    local script_path = clean_string(dt.preferences.read(MOD, "script_path", "string"))

    if python_bin == "" or script_path == "" then
        dt.print_error(_("Deep-Rep: Python binary or script path not configured in preferences."))
        return
    end

    local method_idx = GUI.keyframe_method.selected
    local explicit_kf_name = GUI.keyframe_selector.value
    local keyframe_img, extra_imgs = separate_keyframe_and_extras(selected_images, method_idx, explicit_kf_name)

    local kf_path = get_image_full_path(keyframe_img)
    local kf_cmd_path = normalize_path_for_command(kf_path, python_bin)

    local mod_val = GUI.module.value
    if string.find(mod_val, "denoise") then
        mod_val = "burst_denoise"
    else
        mod_val = "burstsr"
    end

    local scale_str = GUI.sr_scale.value
    local scale_num = "2.0"
    if string.find(scale_str, "1.0") then scale_num = "1.0"
    elseif string.find(scale_str, "4.0") then scale_num = "4.0" end

    local tile_str = GUI.tile_size.value
    local tile_num = "192"
    if string.find(tile_str, "128") then tile_num = "128"
    elseif string.find(tile_str, "256") then tile_num = "256"
    elseif string.find(tile_str, "384") then tile_num = "384"
    elseif string.find(tile_str, "512") then tile_num = "512" end

    -- Build command line arguments
    local python_exec = python_bin
    if not string.find(python_exec, "^wsl") and not string.find(python_exec, '^["]') and string.find(python_exec, " ") then
        python_exec = '"' .. python_exec .. '"'
    end

    local cmd = python_exec .. ' "' .. script_path .. '"'
    cmd = cmd .. ' --keyframe "' .. kf_cmd_path .. '"'

    if #extra_imgs > 0 then
        cmd = cmd .. ' --extra-frames'
        for _, extra_img in ipairs(extra_imgs) do
            local ef_path = get_image_full_path(extra_img)
            local ef_cmd_path = normalize_path_for_command(ef_path, python_bin)
            cmd = cmd .. ' "' .. ef_cmd_path .. '"'
        end
    end

    cmd = cmd .. ' --module ' .. mod_val
    cmd = cmd .. ' --tile-size ' .. tile_num
    cmd = cmd .. ' --sr-scale ' .. scale_num

    if GUI.use_fp16.value then
        cmd = cmd .. ' --fp16'
    else
        cmd = cmd .. ' --no-fp16'
    end

    if headless then
        cmd = cmd .. ' --headless'
        dt.print(_("Deep-Rep: Starting background burst restoration..."))

        local ret = dsys.external_command(cmd)
        if ret == 0 then
            dt.print(_("Deep-Rep: Restoration finished successfully!"))

            -- Determine expected output Linear DNG path
            local kf_dir = keyframe_img.path
            local kf_stem = string.gsub(keyframe_img.filename, "%.%w+$", "")
            local kf_ext = string.match(keyframe_img.filename, "%.%w+$") or ".dng"
            local out_ext = (string.upper(kf_ext) == kf_ext) and ".DNG" or ".dng"
            local sep = is_windows and "\\" or "/"
            local out_file = kf_dir .. sep .. kf_stem .. "_restored" .. out_ext

            if dt.preferences.read(MOD, "auto_import", "bool") and df.check_if_file_exists(out_file) then
                local imported = dt.database.import(out_file)
                if imported then
                    local tag_deeprep = dt.tags.create("deep-rep|restored")
                    dt.tags.attach(tag_deeprep, imported)
                    dt.print(_("Deep-Rep: Imported restored image into filmroll: ") .. kf_stem .. "_restored" .. out_ext)
                end
            end
        else
            dt.print_error(_("Deep-Rep: Process failed with return code ") .. tostring(ret))
        end
    else
        dt.print(_("Deep-Rep: Launching GUI with ") .. tostring(#selected_images) .. _(" images..."))
        if is_windows then
            -- On Windows, launch non-blocking
            dsys.external_command('start "" ' .. cmd)
        else
            dsys.external_command(cmd .. ' &')
        end
    end
end

-- Dynamic refresh of selected images in dropdown
local function update_selection_ui()
    local selected = dt.gui.selection()
    local names = {}
    for _, img in ipairs(selected) do
        table.insert(names, img.filename)
    end
    if #names == 0 then
        names = { _("(no images selected)") }
    end
    GUI.keyframe_selector.items = names
    GUI.keyframe_selector.selected = 1
end

-- GUI Widget Construction
GUI.keyframe_method = dt.new_widget("combobox"){
    label = _("Keyframe Designation:"),
    tooltip = _("Method used to choose the reference keyframe from selected images"),
    selected = 1,
    _("Select from List Below"),
    _("Highest Star Rating"),
    _("Red Color Label"),
    _("First Selected Image")
}

GUI.keyframe_selector = dt.new_widget("combobox"){
    label = _("Keyframe Image:"),
    tooltip = _("Currently selected keyframe reference file"),
    selected = 1,
    _("(no images selected)")
}

GUI.module = dt.new_widget("combobox"){
    label = _("Module:"),
    tooltip = _("Target neural restoration model"),
    selected = 1,
    "burstsr (Super-Resolution)",
    "burst_denoise (Denoise & Color)"
}

GUI.sr_scale = dt.new_widget("combobox"){
    label = _("Output Scale:"),
    tooltip = _("Target resolution scale relative to native sensor grid"),
    selected = 1,
    "2.0x (Balanced - 4x Pixels)",
    "1.0x (Native Resolution - Sharp RGB)",
    "4.0x (Ultra - 16x Pixels)"
}

GUI.tile_size = dt.new_widget("combobox"){
    label = _("Tile Size:"),
    tooltip = _("Inference tile size in pixels (lower sizes use less VRAM)"),
    selected = 1,
    "192 (Recommended - Low VRAM)",
    "128 (Lowest VRAM - 8GB GPU)",
    "256 (Medium VRAM)",
    "384 (High VRAM)",
    "512 (Ultra-High VRAM)"
}

GUI.use_fp16 = dt.new_widget("check_button"){
    label = _("Half-Precision (FP16 Decoder)"),
    tooltip = _("Accelerates decoder and reduces memory usage while preserving full FP32 alignment"),
    value = true
}

GUI.btn_open_gui = dt.new_widget("button"){
    label = _("Open in Deep-Rep GUI"),
    tooltip = _("Forward selected RAW files directly to the standalone Deep-Rep desktop application"),
    clicked_callback = function()
        execute_deeprep(false)
    end
}

GUI.btn_run_batch = dt.new_widget("button"){
    label = _("Process in Background & Import"),
    tooltip = _("Run neural processing headlessly and automatically import the restored DNG into Darktable"),
    clicked_callback = function()
        execute_deeprep(true)
    end
}

local deeprep_box = dt.new_widget("box"){
    orientation = "vertical",
    GUI.keyframe_method,
    GUI.keyframe_selector,
    GUI.module,
    GUI.sr_scale,
    GUI.tile_size,
    GUI.use_fp16,
    GUI.btn_open_gui,
    GUI.btn_run_batch
}

-- Register module into Darktable Lighttable panel
dt.register_lib(
    "deeprep_lib",
    _("Deep-Rep Burst Restoration"),
    true,
    true,
    { [dt.gui.views.lighttable] = { "DT_UI_CONTAINER_PANEL_RIGHT_CENTER", 100 } },
    deeprep_box
)

-- Register event listener to update image dropdown when selection changes
dt.register_event("deeprep_selection", "selection-changed", function()
    update_selection_ui()
end)

return script_data
