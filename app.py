#!/usr/bin/env python
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GdkPixbuf

builder = Gtk.Builder()
builder.add_from_file("app.glade")

win = builder.get_object('window1')
win.about = builder.get_object('aboutdialog1')

imageList = builder.get_object("listbox1")
imageView = builder.get_object('image1')

# imageList.set_sort_func(lambda r1, r2, data, notify_destroy: return -1, None, False)
# imageList.set_filter_func(lambda r, r2, data, notify_destroy: return True, None, False)

# ajouter image dans View
def show_image(name):
    size = imageView.get_allocation()
    pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_size(name, size.width, size.height)
    imageView.set_from_pixbuf(pixbuf)
# ajouter image dans la liste
def add_image(name):
    it = Gtk.ListBoxRow()
    it.data = name
    it.add(Gtk.Label(name.split('/')[-1]))
    imageList.add(it)
def add_images(names):
    for name in names:
        add_image(name)
    imageList.show_all()
    show_image(names[-1])


imageList.connect('row-activated', lambda w, row: show_image(row.data))

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
        chooser.set_select_multiple(True)

        image_filter = Gtk.FileFilter()
        image_filter.set_name("Image files")
        image_filter.add_pixbuf_formats()
        any_filter = Gtk.FileFilter()
        any_filter.set_name("Any files")
        any_filter.add_pattern("*")

        chooser.add_filter(image_filter)
        chooser.add_filter(any_filter)
        response = chooser.run()
        if response == Gtk.ResponseType.OK:
            add_images(chooser.get_filenames())
        chooser.destroy()

    def on_about_clicked(self, *args):
        win.about.show_all()
        # .run
        # .destroy

    def on_about_closed(self, *args):
        print("on_about_closed")
        win.about.hide()

builder.connect_signals(EventHandler())
win.show_all()
Gtk.main()
