using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Net;
using System.Net.Cache;
using System.Reactive.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace checkface_dotnet
{
    public partial class CheckFaceForm : Form
    {
        public CheckFaceForm(string value = "", FileHasher hasher = null)
        {
            InitializeComponent();
            textBox1.Text = value;
            WebRequest.DefaultCachePolicy
                = new HttpRequestCachePolicy(HttpRequestCacheLevel.CacheIfAvailable);
            var changes = System.Reactive.Linq.Observable.FromEventPattern(h => textBox1.TextChanged += h, r => textBox1.TextChanged -= r);
            var subscription = changes.Throttle(TimeSpan.FromMilliseconds(250)).Subscribe(_ =>
            {
                newValue();
            });
            pictureBox1.LoadCompleted += (s, e) => pictureBox1.SizeMode = PictureBoxSizeMode.Zoom;
            this.FormClosing += (s, e) => subscription.Dispose();
            this.Hasher = hasher;
        }

        public FileHasher Hasher { get; }

        protected override void OnLoad(EventArgs e)
        {
            base.OnLoad(e);
            if(Hasher != null)
            {
                textBox1.Text = Hasher.HashFile(textBox1.Text);
            }
            newValue();
        }

        private void newValue()
        {
            pictureBox1.CancelAsync();
            if (String.IsNullOrWhiteSpace(textBox1.Text))
            {
                pictureBox1.Image = Properties.Resources.face_transparent;
            }
            else
            {
                pictureBox1.Image = Properties.Resources.loader;
                pictureBox1.SizeMode = PictureBoxSizeMode.CenterImage;
                string enc = System.Web.HttpUtility.UrlEncode(textBox1.Text);
                pictureBox1.LoadAsync($"https://api.checkface.ml/api/face/?value={enc}&dim=300");
            }
        }
    }
}
