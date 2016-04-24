# -*- coding: utf-8 -*-
from gi.repository import Gtk, Gdk, GdkPixbuf
from .core import ImageAnalyzer


class EventHandler():
    """Signal Event handlers definition"""

    def __init__(self, app):
        self.app = app

    def on_quit_clicked(self, *args):
        """clean and close the app"""
        Gtk.main_quit(*args)

    def on_clear_clicked(self, *args):
        """clear images list and image view
        recuperation of default value from graphical object
        """
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
        self.app.beta.set_value(0.1)
        self.app.sigmah.set_value(0.01)
        self.app.vh.set_value(0.1)
        self.app.dt.set_value(1)
        self.app.thrf.set_value(4)
        self.app.tr.set_value(1)
        self.app.k.set_value(2)
        self.app.m.set_value(1)
        self.app.nitmin.set_value(30)
        self.app.nitmax.set_value(30)
        self.app.scale.set_value(1)
        self.app.pl.set_active(True)
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
                                     facteur=float(self.app.facteur.get_text()))
        img_analyzer.lecture_data(self.app.xmin.get_value_as_int(),
                                  self.app.xmax.get_value_as_int(),
                                  self.app.ymin.get_value_as_int(),
                                  self.app.ymax.get_value_as_int())
        img_analyzer.post_lecture()
        img_analyzer.init_params(beta=self.app.beta.get_value(),
                                 sigmaH=self.app.sigmah.get_value(),
                                 v_h_facture=self.app.vh.get_value_as_int(),
                                 dt=self.app.dt.get_value_as_int(),
                                 Thrf=self.app.thrf.get_value_as_int(),
                                 TR=self.app.tr.get_value_as_int(),
                                 K=self.app.k.get_value_as_int(),
                                 M=self.app.m.get_value_as_int(),
                                 )
        img_analyzer.set_flags(pl=1 if self.app.pl.get_active() else 0)
        fgs1 = img_analyzer.gen_hrf(nItMin=self.app.nitmin.get_value_as_int(),
                                    nItMax=self.app.nitmax.get_value_as_int(),
                                    scale=self.app.scale.get_value_as_int(),
                                    )
        self.app.add_result('fonction de réponse', fgs1[0])
        self.app.add_result('Mélange à posteriori', fgs1[1])
        fgs2 = img_analyzer.gen_nrl()
        self.app.add_result('Niveau de réponse', fgs2[0])
        self.app.add_result('Label activation', fgs2[1])
        self.app.notebook.set_current_page(2)
        # i = 0
        # for fig in fgs1:
        # self.app.add_result('fonction de reponse' , fig)
        # self.app.add_result('Mélange a posteriori ' , fig)
        # i += 1
        # i = 0
        # for fig in fgs2:
        # self.app.add_result('nrl' + str(i), fig)
        # i += 1

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

    def on_facteur_ok(self, *args):
        print("Clicked")
        self.app.facteur.set_text(str(self.app.spinbutton_facteur.get_value_as_int()))
        self.app.window_facteur.hide()
