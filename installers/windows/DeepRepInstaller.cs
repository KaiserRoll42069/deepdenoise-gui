// Deep-Rep Standalone Windows Installer
// Original Deep-Rep Framework Copyright (c) 2021 Huawei Technologies Co., Ltd.
// Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International)
// https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
// Academic / non-commercial research use only.

using System;
using System.IO;
using System.Drawing;
using System.Diagnostics;
using System.Threading;
using System.Windows.Forms;

namespace DeepRepInstaller
{
    public class InstallerForm : Form
    {
        private Label lblHeader;
        private Label lblDiskSpace;
        private Label lblPython;
        private Label lblTorch;
        private CheckBox chkInstallTorch;
        private CheckBox chkDarktable;
        private CheckBox chkWeights;
        private Button btnInstall;
        private Button btnLaunch;
        private TextBox txtLog;
        private ProgressBar progressBar;
        private string detectedPython = null;
        private double freeDiskGB = 0;

        public InstallerForm()
        {
            InitializeComponent();
            CheckSystemStatus();
        }

        private void InitializeComponent()
        {
            this.Text = "Deep-Rep RAW Burst Restoration - Windows Installer";
            this.Size = new Size(680, 560);
            this.StartPosition = FormStartPosition.CenterScreen;
            this.FormBorderStyle = FormBorderStyle.FixedDialog;
            this.MaximizeBox = false;
            this.Font = new Font("Segoe UI", 9F, FontStyle.Regular, GraphicsUnit.Point);
            this.BackColor = Color.FromArgb(248, 249, 250);

            lblHeader = new Label()
            {
                Text = "Deep-Rep Toolkit Windows Setup",
                Font = new Font("Segoe UI", 14F, FontStyle.Bold, GraphicsUnit.Point),
                ForeColor = Color.FromArgb(33, 37, 41),
                Location = new Point(20, 15),
                Size = new Size(630, 30)
            };

            lblDiskSpace = new Label()
            {
                Text = "Checking disk space...",
                Location = new Point(25, 55),
                Size = new Size(620, 22),
                Font = new Font("Segoe UI", 9.5F, FontStyle.Regular, GraphicsUnit.Point)
            };

            lblPython = new Label()
            {
                Text = "Checking Python installation...",
                Location = new Point(25, 80),
                Size = new Size(620, 22),
                Font = new Font("Segoe UI", 9.5F, FontStyle.Regular, GraphicsUnit.Point)
            };

            lblTorch = new Label()
            {
                Text = "Checking PyTorch...",
                Location = new Point(25, 105),
                Size = new Size(620, 22),
                Font = new Font("Segoe UI", 9.5F, FontStyle.Regular, GraphicsUnit.Point)
            };

            chkInstallTorch = new CheckBox()
            {
                Text = "Install/Update PyTorch with CUDA 12.1 (NVIDIA GPU Acceleration)",
                Location = new Point(25, 135),
                Size = new Size(620, 24),
                Checked = false
            };

            chkWeights = new CheckBox()
            {
                Text = "Download Pretrained Neural Models (BURST-SR, Denoise, PWCNet)",
                Location = new Point(25, 160),
                Size = new Size(620, 24),
                Checked = true
            };

            chkDarktable = new CheckBox()
            {
                Text = "Configure Darktable Integration (Install deeprep.lua plugin)",
                Location = new Point(25, 185),
                Size = new Size(620, 24),
                Checked = true
            };

            btnInstall = new Button()
            {
                Text = "Start Installation / Verification",
                Location = new Point(25, 218),
                Size = new Size(240, 36),
                Font = new Font("Segoe UI", 9.5F, FontStyle.Bold, GraphicsUnit.Point),
                BackColor = Color.FromArgb(13, 110, 253),
                ForeColor = Color.White,
                FlatStyle = FlatStyle.Flat,
                Cursor = Cursors.Hand
            };
            btnInstall.FlatAppearance.BorderSize = 0;
            btnInstall.Click += BtnInstall_Click;

            btnLaunch = new Button()
            {
                Text = "Launch Deep-Rep GUI",
                Location = new Point(280, 218),
                Size = new Size(180, 36),
                Font = new Font("Segoe UI", 9.5F, FontStyle.Regular, GraphicsUnit.Point),
                Enabled = false,
                Cursor = Cursors.Hand
            };
            btnLaunch.Click += (s, e) =>
            {
                string baseDir = AppDomain.CurrentDomain.BaseDirectory;
                string launcher = Path.Combine(baseDir, "DeepRepLauncher.exe");
                if (File.Exists(launcher)) Process.Start(launcher);
                else
                {
                    string bat = Path.Combine(baseDir, "run_deeprep.bat");
                    if (File.Exists(bat)) Process.Start(bat);
                }
            };

            progressBar = new ProgressBar()
            {
                Location = new Point(25, 265),
                Size = new Size(615, 16),
                Style = ProgressBarStyle.Blocks
            };

            txtLog = new TextBox()
            {
                Location = new Point(25, 290),
                Size = new Size(615, 215),
                Multiline = true,
                ScrollBars = ScrollBars.Vertical,
                ReadOnly = true,
                BackColor = Color.FromArgb(30, 30, 30),
                ForeColor = Color.FromArgb(220, 220, 220),
                Font = new Font("Consolas", 8.5F, FontStyle.Regular, GraphicsUnit.Point)
            };

            this.Controls.Add(lblHeader);
            this.Controls.Add(lblDiskSpace);
            this.Controls.Add(lblPython);
            this.Controls.Add(lblTorch);
            this.Controls.Add(chkInstallTorch);
            this.Controls.Add(chkWeights);
            this.Controls.Add(chkDarktable);
            this.Controls.Add(btnInstall);
            this.Controls.Add(btnLaunch);
            this.Controls.Add(progressBar);
            this.Controls.Add(txtLog);
        }

        private void CheckSystemStatus()
        {
            try
            {
                DriveInfo cDrive = new DriveInfo("C");
                freeDiskGB = Math.Round((double)cDrive.AvailableFreeSpace / (1024 * 1024 * 1024), 2);
                if (freeDiskGB < 4.0)
                {
                    lblDiskSpace.Text = string.Format("Disk Space: {0} GB free on C: (WARNING: Minimum 4.0 GB required)", freeDiskGB);
                    lblDiskSpace.ForeColor = Color.Red;
                    btnInstall.Enabled = false;
                }
                else
                {
                    lblDiskSpace.Text = string.Format("Disk Space: {0} GB free on C: (Healthy, >= 4.0 GB preserved)", freeDiskGB);
                    lblDiskSpace.ForeColor = Color.DarkGreen;
                }

                // Check Python
                detectedPython = FindPython();
                if (!string.IsNullOrEmpty(detectedPython))
                {
                    lblPython.Text = "Python: Found at " + detectedPython;
                    lblPython.ForeColor = Color.DarkGreen;
                    CheckTorch(detectedPython);
                }
                else
                {
                    lblPython.Text = "Python: Not detected. Please install Python 3.10-3.12 from python.org.";
                    lblPython.ForeColor = Color.OrangeRed;
                    lblTorch.Text = "PyTorch: Waiting for Python installation.";
                }

                // Enable launch if files exist
                string baseDir = AppDomain.CurrentDomain.BaseDirectory;
                if (File.Exists(Path.Combine(baseDir, "deeprep_gui.py")) && !string.IsNullOrEmpty(detectedPython))
                {
                    btnLaunch.Enabled = true;
                }
            }
            catch (Exception ex)
            {
                AppendLog("Status check error: " + ex.Message);
            }
        }

        private void CheckTorch(string pyExe)
        {
            try
            {
                ProcessStartInfo psi = new ProcessStartInfo()
                {
                    FileName = pyExe,
                    Arguments = "-c \"import torch; print(f'{torch.__version__}|{torch.cuda.is_available()}|{torch.cuda.get_device_name(0) if torch.cuda.is_available() else \\\"CPU\\\"}')\"",
                    RedirectStandardOutput = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                };
                using (Process p = Process.Start(psi))
                {
                    string output = p.StandardOutput.ReadToEnd().Trim();
                    p.WaitForExit();
                    if (p.ExitCode == 0 && output.Contains("|"))
                    {
                        string[] parts = output.Split('|');
                        bool cuda = parts[1].ToLower().Contains("true");
                        lblTorch.Text = string.Format("PyTorch: v{0} ({1} - {2})", parts[0], cuda ? "CUDA Accelerated" : "CPU Only", parts[2]);
                        lblTorch.ForeColor = cuda ? Color.DarkGreen : Color.DarkGoldenrod;
                        chkInstallTorch.Checked = false;
                        chkInstallTorch.Text = "Reinstall / Upgrade PyTorch with CUDA 12.1";
                    }
                    else
                    {
                        lblTorch.Text = "PyTorch: Not currently installed in Python environment.";
                        lblTorch.ForeColor = Color.DarkOrange;
                        chkInstallTorch.Checked = true;
                    }
                }
            }
            catch
            {
                lblTorch.Text = "PyTorch: Could not verify.";
            }
        }

        private string FindPython()
        {
            string localAppData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
            string[] candidates = new string[]
            {
                Path.Combine(localAppData, @"Programs\Python\Python312\python.exe"),
                Path.Combine(localAppData, @"Programs\Python\Python311\python.exe"),
                Path.Combine(localAppData, @"Programs\Python\Python310\python.exe"),
                @"C:\Program Files\Python312\python.exe",
                @"C:\Program Files\Python311\python.exe",
                @"C:\Program Files\Python310\python.exe"
            };

            foreach (string p in candidates)
            {
                if (File.Exists(p)) return p;
            }

            string pathEnv = Environment.GetEnvironmentVariable("PATH");
            if (!string.IsNullOrEmpty(pathEnv))
            {
                foreach (string dir in pathEnv.Split(';'))
                {
                    string t = dir.Trim();
                    if (!string.IsNullOrEmpty(t))
                    {
                        string test = Path.Combine(t, "python.exe");
                        if (File.Exists(test) && !test.ToLower().Contains("windowsapps")) return test;
                    }
                }
            }
            return null;
        }

        private void BtnInstall_Click(object sender, EventArgs e)
        {
            if (freeDiskGB < 4.0)
            {
                MessageBox.Show("Cannot proceed: Free disk space on C: is below 4.0 GB.", "Disk Space Safety Check", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }

            btnInstall.Enabled = false;
            progressBar.Style = ProgressBarStyle.Marquee;
            AppendLog("Starting Deep-Rep Windows Installation & Environment Verification...\n");

            Thread thread = new Thread(RunInstallWorker) { IsBackground = true };
            thread.Start();
        }

        private void RunInstallWorker()
        {
            try
            {
                string baseDir = AppDomain.CurrentDomain.BaseDirectory;
                string ps1 = Path.Combine(baseDir, "install_windows.ps1");
                if (!File.Exists(ps1))
                {
                    AppendLog("ERROR: Cannot find install_windows.ps1 in: " + baseDir);
                    SetUIFinished(false);
                    return;
                }

                string args = string.Format("-NoProfile -ExecutionPolicy Bypass -File \"{0}\" -NonInteractive", ps1);
                if (chkInstallTorch.Checked)
                {
                    args += " -InstallTorch";
                }

                ProcessStartInfo psi = new ProcessStartInfo()
                {
                    FileName = "powershell.exe",
                    Arguments = args,
                    WorkingDirectory = baseDir,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                };

                using (Process proc = new Process() { StartInfo = psi })
                {
                    proc.OutputDataReceived += (s, ev) => { if (ev.Data != null) AppendLog(ev.Data); };
                    proc.ErrorDataReceived += (s, ev) => { if (ev.Data != null) AppendLog("WARN: " + ev.Data); };

                    proc.Start();
                    proc.BeginOutputReadLine();
                    proc.BeginErrorReadLine();
                    proc.WaitForExit();

                    bool success = (proc.ExitCode == 0);
                    SetUIFinished(success);
                }
            }
            catch (Exception ex)
            {
                AppendLog("Installation Exception: " + ex.Message);
                SetUIFinished(false);
            }
        }

        private void SetUIFinished(bool success)
        {
            if (this.InvokeRequired)
            {
                this.Invoke(new Action<bool>(SetUIFinished), success);
                return;
            }

            progressBar.Style = ProgressBarStyle.Blocks;
            progressBar.Value = success ? 100 : 0;
            btnInstall.Enabled = true;
            btnLaunch.Enabled = success || !string.IsNullOrEmpty(detectedPython);

            if (success)
            {
                AppendLog("\n[SUCCESS] Installation & Environment Configuration Complete!");
                MessageBox.Show("Deep-Rep setup has completed successfully!", "Deep-Rep Installed", MessageBoxButtons.OK, MessageBoxIcon.Information);
                CheckSystemStatus();
            }
            else
            {
                AppendLog("\n[ERROR] Setup encountered errors. Please check the log messages above.");
            }
        }

        private void AppendLog(string text)
        {
            if (txtLog.InvokeRequired)
            {
                txtLog.Invoke(new Action<string>(AppendLog), text);
                return;
            }
            txtLog.AppendText(text + Environment.NewLine);
        }

        [STAThread]
        static void Main(string[] args)
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new InstallerForm());
        }
    }
}
