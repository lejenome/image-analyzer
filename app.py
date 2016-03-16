import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GdkPixbuf

builder = Gtk.Builder()
builder.add_from_file("app.glade")

image_filter = builder.get_object('filefilter1')
image_filter.set_name("Image files")
image_filter.add_pixbuf_formats()
any_filter = builder.get_object('filefilter2')
any_filter.set_name("Any files")
any_filter.add_mime_type("*")

win = builder.get_object('window1')
win.about = builder.get_object('aboutdialog1')

imageList = builder.get_object("listbox1")
imageView = builder.get_object('image1')

# ajouter image dans View
def showImage(name):
    size = imageView.get_allocation()
    pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_size(name, size.width, size.height)
    imageView.set_from_pixbuf(pixbuf)
# ajouter image dans la liste
def addImage(name):
    it = Gtk.ListBoxRow()
    it.data = name
    it.add(Gtk.Label(name.split('/')[-1]))
    imageList.add(it)
    imageList.show_all()
    showImage(name)

imageList.connect('row-activated', lambda w, row: showImage(row.data))

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
        win.about.show_all()

builder.connect_signals(EventHandler())
win.show_all()
Gtk.main()
