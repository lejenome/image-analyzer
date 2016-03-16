import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk

# class AppWindow(Gtk.Window):
#    def __init__(self):
#        Gtk.Window.__init__(self, title="Picture Analyzer")
#
#        self.root = Gtk.Box(spacing=6)
#        self.add(self.root)
#
#        self.lbl1 = Gtk.Label(label="lbl1", angle=25, halign=Gtk.Align.END)
#        self.root.pack_start(self.lbl1, True, True, 0)
#        self.btn1 = Gtk.Button(label="Btn1")
#        self.btn1.connect('clicked', self.on_btn1_clicked)
#        self.root.pack_end(self.btn1, True, True, 0)
#
#    def on_btn1_clicked(self, widget):
#        print('hi')

builder = Gtk.Builder()
builder.add_from_file("app.glade")

image_filter = builder.get_object('filefilter1')
image_filter.set_name("Image files")
image_filter.add_mime_type("image/*")
any_filter = builder.get_object('filefilter2')
any_filter.set_name("Any files")
any_filter.add_mime_type("*")

win = builder.get_object('window1')
about = builder.get_object('aboutdialog1')

imageList = builder.get_object("listbox1")
imageView = builder.get_object('image1')

def addImage(name):
    it = Gtk.ListBoxRow()
    it.data = name
    it.add(Gtk.Label(name.split('/')[-1]))
    imageList.add(it)
    imageList.show_all()
    imageView.set_from_file(name)

imageList.connect('row-activated', lambda w, row: imageView.set_from_file(row.data))

class EventHandler():

    def on_quit_clicked(self, *args):
        Gtk.main_quit(*args)

    def on_clear_clicked(self, *args):
        imageView.clear()

    def on_add_clicked(self, *args):
        chooser = Gtk.FileChooserDialog("Choose an image", win,
                                        Gtk.FileChooserAction.OPEN,
                                        (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                                         Gtk.STOCK_OPEN, Gtk.ResponseType.OK))
        chooser.add_filter(image_filter)
        chooser.add_filter(any_filter)
        response = chooser.run()
        if response == Gtk.ResponseType.OK:
            addImage(chooser.get_filename())
        chooser.destroy()

    def on_about_clicked(self, *args):
        about.show_all()

# win = AppWindow()
# win.connect('delete-event', Gtk.main_quit)

builder.connect_signals(EventHandler())
win.show_all()
Gtk.main()
