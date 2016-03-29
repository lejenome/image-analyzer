#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Gtk Interface for Image Analyzer using glade template"""

import os
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GdkPixbuf
from matplotlib.backends.backend_gtk3cairo import FigureCanvasGTK3Cairo as FigureCanvas

# imageList.set_sort_func(lambda r1, r2, data, notify_destroy: return -1, None, False)
# imageList.set_filter_func(lambda r, r2, data, notify_destroy: return True, None, False)


class EventHandler():
    """Signal Event handlers definition"""

    def __init__(self, app):
        self.app = app

    def on_quit_clicked(self, *args):
        """clean and close the app"""
        Gtk.main_quit(*args)

    def on_clear_clicked(self, *args):
        """clear images list and image view"""
        self.app.imageView.clear()

    def on_add_clicked(self, *args):
        """Launch multi-select image file chooser dialog and append new files
        to the image list and show last selected file"""
        chooser = Gtk.FileChooserDialog("Choose an image", self.app.win,
                                        Gtk.FileChooserAction.OPEN,
                                        (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                                         Gtk.STOCK_OPEN, Gtk.ResponseType.OK))
        chooser.set_select_multiple(True)

        image_filter = Gtk.FileFilter()
        image_filter.set_name("Image files")
        image_filter.add_pixbuf_formats()
        # any_filter = Gtk.FileFilter()
        # any_filter.set_name("Any files")
        # any_filter.add_pattern("*")

        chooser.add_filter(image_filter)
        # chooser.add_filter(any_filter)
        response = chooser.run()
        if response == Gtk.ResponseType.OK:
            self.app.add_images(chooser.get_filenames())
        chooser.destroy()

    def on_about_clicked(self, *args):
        """show about dialog"""
        self.app.win.about.show_all()
        # .run
        # .destroy

    def on_about_closed(self, *args):
        print("on_about_closed")
        self.app.win.about.hide()


class App:
    """main logic for the graphical interface

    Usage::

        app = App()
        app.run()
    """

    def __init__(self):
        self.builder = Gtk.Builder()

        glade_file = os.path.join(os.path.dirname(__file__), 'app.glade')
        self.builder.add_from_file(glade_file)

        self.win = self.builder.get_object('window1')
        self.win.about = self.builder.get_object('aboutdialog1')

        self.imageList = self.builder.get_object("listbox1")
        self.imageView = self.builder.get_object('image1')
        self.imageList.connect('row-activated', lambda w, row: self.show_image(row.data))

    def run(self):
        """connect signals and run Gtk window"""
        self.builder.connect_signals(EventHandler(self))
        self.win.show_all()
        Gtk.main()

    # ajouter image dans View
    def show_image(self, name):
        """Show image on image view

        :param name: image file path
        :type name: str
        """
        size = self.imageView.get_allocation()
        pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_size(name, size.width, size.height)
        self.imageView.set_from_pixbuf(pixbuf)

    # ajouter image dans la liste
    def add_image(self, name):
        """Add image to image list

        :param name: image file path
        :type name: str
        """
        it = Gtk.ListBoxRow()
        it.data = name
        it.add(Gtk.Label(name.split('/')[-1]))
        self.imageList.add(it)

    def add_images(self, names):
        """Add images to image list then show last one

        :param names: image files path list
        :type names: array of str
        """
        for name in names:
            self.add_image(name)
        self.imageList.show_all()
        self.show_image(names[-1])
