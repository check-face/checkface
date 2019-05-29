using SharpShell.Attributes;
using SharpShell.SharpContextMenu;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace checkface_dotnet
{
    [ComVisible(true)]
    [COMServerAssociation(AssociationType.AllFiles)]
    public class CheckFaceExtension : SharpContextMenu
    {
        protected override bool CanShowMenu() =>
            SelectedItemPaths.Count() == 1 && File.Exists(SelectedItemPaths.Single());

        protected override ContextMenuStrip CreateMenu()
        {
            //  Create the menu strip.
            var menu = new ContextMenuStrip();

            //  Create a 'count lines' item.
            var checkFaceItem = new ToolStripMenuItem
            {
                Text = "Check Face",
                Image = Properties.Resources.faceico16
            };

            var filename = SelectedItemPaths.Single();
            foreach (var alg in FileHasher.Algorithms)
            {
                var item = new ToolStripMenuItem
                {
                    Text = alg.Name
                };
                checkFaceItem.DropDownItems.Add(item);
                item.Click += (sender, args) => new CheckFaceForm(filename, new FileHasher(alg)).Show();
            }

            //  When we click, we'll call the 'CountLines' function.
            checkFaceItem.DoubleClick += (sender, args)
                => new CheckFaceForm(filename, new FileHasher(HashAlgorithmName.SHA256)).Show();
            checkFaceItem.DoubleClickEnabled = true;

            menu.Items.Add(checkFaceItem);

            //  Return the menu.
            return menu;
        }
    }
}
