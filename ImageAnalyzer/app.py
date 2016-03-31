#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Gtk Interface for Image Analyzer using glade template"""

import os
from matplotlib import pyplot
from tifffile import imread, imshow, TiffFile
from matplotlib.backends.backend_gtk3cairo import FigureCanvasGTK3Cairo as FigureCanvas
from matplotlib.backends.backend_gtk3 import NavigationToolbar2GTK3 as NavigationToolbar
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GdkPixbuf

from ImageAnalyzer.core import ImageAnalyzer

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
        while self.app.imageList.get_row_at_index(0):
            self.app.imageList.get_row_at_index(0).destroy()
        while self.app.resultList.get_row_at_index(0):
            self.app.resultList.get_row_at_index(0).destroy()
        old_viewport = self.app.imageScrolled.get_child()
        if old_viewport:
            old_viewport.destroy()
        old_viewport = self.app.resultScrolled.get_child()
        if old_viewport:
            old_viewport.destroy()
        self.app.xmin.set_value(2658)
        self.app.xmax.set_value(2730)
        self.app.ymin.set_value(2600)
        self.app.ymax.set_value(2680)
        self.app.notebook.set_current_page(0)

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
        image_filter.add_pattern("*.tiff")
        image_filter.add_pattern("*.TIIF")
        image_filter.add_pattern("*.TIF")
        image_filter.add_pattern("*.tif")
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
        """close about dialog"""
        self.app.win.about.hide()

    def on_search_changed(self, *args):
        self.app.imageList.invalidate_filter()

    def on_exec_clicked(self, *args):
        """analyser les images importées
        """
        while self.app.resultList.get_row_at_index(0):
            self.app.resultList.get_row_at_index(0).destroy()
        old_viewport = self.app.resultScrolled.get_child()
        if old_viewport:
            old_viewport.destroy()

        imgs = []
        i = 0

        while self.app.imageList.get_row_at_index(i):
            imgs.append(self.app.imageList.get_row_at_index(i).data)
            i += 1
        img_analyzer = ImageAnalyzer(sorted(imgs),
                                     bande=float(self.app.bande.get_active_text()),
                                     facteur=float(self.app.facteur.get_active_text()))
        img_analyzer.lecture_data(self.app.xmin.get_value_as_int(),
                                  self.app.xmax.get_value_as_int(),
                                  self.app.ymin.get_value_as_int(),
                                  self.app.ymax.get_value_as_int())
        img_analyzer.post_lecture()
        img_analyzer.init_params()
        img_analyzer.set_flags(shower=0)
        fgs1 = img_analyzer.gen_hrf()
        fgs2 = img_analyzer.gen_nrl()

        i = 0
        for fig in fgs1:
            self.app.add_result('hrf' + str(i), fig)
            i += 1
        i = 0
        for fig in fgs2:
            self.app.add_result('nrl' + str(i), fig)
            i += 1
        self.app.notebook.set_current_page(2)

    def on_item_delete(self, widget, ev, *args):
        if ev.keyval == Gdk.KEY_Delete:
            r = self.app.imageList.get_selected_row()
            if r:
                r.destroy()

    def on_xmin_changed(self, *args):
        print("changed")
        self.app.xmax.set_range(self.app.xmin.get_value_as_int() + 1, self.app.shape[0])

    def on_xmax_changed(self, *args):
        print("changed")
        self.app.xmin.set_range(0, self.app.xmax.get_value_as_int() - 1)

    def on_ymin_changed(self, *args):
        print("changed")
        self.app.ymax.set_range(self.app.ymin.get_value_as_int() + 1, self.app.shape[1])

    def on_ymax_changed(self, *args):
        print("changed")
        self.app.ymin.set_range(0, self.app.ymax.get_value_as_int() - 1)


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
        self.imageScrolled = self.builder.get_object('scrolledwindow2')
        self.imageList.connect('row-activated', lambda w, row: self.show_image(row.data))

        self.resultList = self.builder.get_object("listbox2")
        self.resultScrolled = self.builder.get_object('scrolledwindow1')
        self.resultList.connect('row-activated', lambda w, row: self.show_result(row.data))

        self.search = self.builder.get_object('searchentry1')
        self.imageList.set_filter_func(self.filter_images, self.search)

        self.xmin = self.builder.get_object('xmin2')
        self.xmax = self.builder.get_object('xmax2')
        self.ymin = self.builder.get_object('ymin2')
        self.ymax = self.builder.get_object('ymax2')
        self.facteur = self.builder.get_object('facteur')
        self.bande = self.builder.get_object('bande')

        self.notebook = self.builder.get_object('notebook1')
        self.imageBox = self.builder.get_object('scrolledwindow5')
        self.resultBox = self.builder.get_object('scrolledwindow6')

        self.sigmah = self.builder.get_object('sigmah')
        self.beta = self.builder.get_object('beta')
        self.thrf = self.builder.get_object('thrf')
        self.vh = self.builder.get_object('vh')
        self.dt = self.builder.get_object('dt')
        self.tr = self.builder.get_object('tr')
        self.nitmin = self.builder.get_object('nitmin')
        self.nitmax = self.builder.get_object('nitmax')
        self.m = self.builder.get_object('m')
        self.k = self.builder.get_object('k')
        self.scale = self.builder.get_object('scale')
        self.pl = self.builder.get_object('pl')
        self.shape = (0, 5000)

    def run(self):
        """connect signals and run Gtk window"""
        self.builder.connect_signals(EventHandler(self))
        self.win.show_all()
        Gtk.main()

    def filter_images(self, row, search):
        if not search.get_text():
            return True
        else:
            return search.get_text().strip().lower() in row.data

    # ajouter image dans View
    def show_image(self, name):
        """Show image on image view

        :param name: image file path
        :type name: str
        """
        # size = self.imageView.get_allocation()
        # pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_size(name, size.width, size.height)
        # self.imageView.set_from_pixbuf(pixbuf)
        old_viewport = self.imageScrolled.get_child()
        if old_viewport:
            old_viewport.destroy()
        old_viewport = self.imageBox.get_child()
        if old_viewport:
            old_viewport.destroy()
        with TiffFile(name) as img:
            fig = imshow(img.asarray())[0]
            canvas = FigureCanvas(fig)
            self.imageScrolled.add_with_viewport(canvas)
            toolbar = NavigationToolbar(canvas, self.win)
            self.imageBox.add_with_viewport(toolbar)
            pyplot.close(fig)

            self.shape = img.asarray().shape
            self.xmax.set_range(self.xmin.get_value_as_int() + 1, self.shape[0])
            self.ymax.set_range(self.ymin.get_value_as_int() + 1, self.shape[1])

        self.imageScrolled.show_all()

    # ajouter image dans la liste
    def add_image(self, name):
        """Add image to image list

        :param name: image file path
        :type name: str
        """

        i = 0
        while self.imageList.get_row_at_index(i):
            if self.imageList.get_row_at_index(i).data == name:
                return
            i += 1

        it = Gtk.ListBoxRow()
        it.data = name
        name_shorten = name.split('/')[-1]
        name_shorten = name_shorten.split('.')[0]
        try:
            name_shorten = name_shorten.split('_')[3]
        except:
            pass
        it.add(Gtk.Label(name_shorten))
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

    def add_result(self, name, fig):
        """ajouter un nouveau row avec le nom de l'image et enregister l'object fig on attr data

        :param name: chemin du fichier image à ajouter
        :type name: str
        :param fig: l'objet figure genereté par imread
        :type fig: matplotlib.figure.Figure
        """
        it = Gtk.ListBoxRow()
        it.data = fig
        it.add(Gtk.Label(name))
        self.resultList.add(it)
        self.resultList.show_all()

    def show_result(self, fig):
        old_viewport = self.resultScrolled.get_child()
        if old_viewport:
            old_viewport.destroy()
        old_viewport = self.resultBox.get_child()
        if old_viewport:
            old_viewport.destroy()
        canvas = FigureCanvas(fig)
        self.resultScrolled.add_with_viewport(canvas)
        toolbar = NavigationToolbar(canvas, self.win)
        self.resultBox.add_with_viewport(toolbar)
        self.resultScrolled.show_all()
