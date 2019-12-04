import PySimpleGUI as sg

def get_browser_layout():
    #Window One Design
    #--------------------------------------------------------#
    sg.ChangeLookAndFeel('Material1')

    # ------ Menu Definition ------ #
    menu_def = [['File', ['Exit']]]

    # ------ Column Definition ------ #
    column1 = [
        [
            sg.Text('Column 1',
                    background_color='#F7F3EC',
                    justification='center',
                    size=(10, 1))
        ],
        [sg.Spin(values=('Spin Box 1', '2', '3'), initial_value='Spin Box 1')],
        [sg.Spin(values=('Spin Box 1', '2', '3'), initial_value='Spin Box 2')],
        [sg.Spin(values=('Spin Box 1', '2', '3'), initial_value='Spin Box 3')]
    ]

    layout = [
        [sg.Menu(menu_def, tearoff=True)],
        [
            sg.Text('Seam Carving - content-aware image resizing',
                    size=(40, 1),
                    justification='center',
                    font=("Helvetica", 24),
                    relief=sg.RELIEF_RIDGE)
        ],
            # ------ Folder Selector ------ #
        [
            sg.Text(
                'Choose A File',
                size=(100, 1),
                auto_size_text=True,
                justification='left')
        #         tooltip=
        #         'Directory Selection. Defaults to current Working Directory.'),
        #     sg.InputText(directory, size=(80, 1)),
            #sg.FolderBrowse()
        ],
        
        [sg.FileBrowse(target=(-1, 0))],

        #  # ------ Image Selector ------ #
        # [
        #     sg.Text(
        #         'Choose An Image',
        #         size=(20, 1),
        #         auto_size_text=True,
        #         justification='center',
        #         tooltip=
        #         'Image Selection. Gui can only display .png files. jpgs run through conversion apriori.'
        #     ),
        #     sg.InputCombo(([i for i in files]),
        #                   default_value=files[0],
        #                   size=(80, 1))
        # ],
        [
            sg.Frame(layout=[
                # ------ Dimension Selector ------ #
                [
                    sg.Text('Choose A Dim To Carve Along',
                            size=(25, 1),
                            auto_size_text=True,
                            justification='center',
                            tooltip='Row or Column Carving'),
                    sg.InputCombo(('Column', 'Row', 'Both'),
                                    default_value='Column',
                                    size=(20, 1))
                ],
                # ------ Filter Selector ------ #

                [
                    sg.Text('Choose A Filter To Use',
                            size=(25, 1),
                            auto_size_text=True,
                            justification='center',
                            tooltip='Filter Selection'),
                    sg.InputCombo(('Sobel', 'Sobel_Feldman', 'Scharr'),
                                    default_value='Sobel',
                                    size=(20, 1))
                ],
                # ------ Rescale Size Selector ------ #
                [
                    sg.Text('Rescaling Factor',
                            size=(25, 1),
                            auto_size_text=True,
                            justification='center',
                            tooltip='Filter Selection'),
                    sg.Slider(range=(1, 100),
                                orientation='h',
                                size=(34, 20),
                                default_value=98)
                ],
                # ------ Save Progress Slider ------ #
                [
                    sg.Text('Save Every K Seams',
                            size=(25, 1),
                            auto_size_text=True,
                            justification='center',
                            tooltip='Filter Selection'),
                    sg.Slider(range=(1, 20),
                                orientation='h',
                                size=(34, 20),
                                default_value=20)
                ],
            ],
                        title='Options',
                        title_color='red',
                        relief=sg.RELIEF_SUNKEN,
                        tooltip='Set Parameters to Feed into SC algo')
        ],
        [sg.Text('_' * 80)],
        [sg.Button('Launch'),
            sg.Cancel()],
    ]

    return layout





def get_carved_image_layout(image_path, output):
    layout = [
                #[sg.PopupAnimated('gif\movie.gif')],
                [
                    sg.Text('Original Image'),
                    sg.Image(r"{}".format(image_path)),
                    sg.Image(r"{}".format(output)),
                    sg.Text('Carved Image'),

                ],
                [sg.Image(r"{}".format('../energy/energymap.png'))],
                [sg.Button('Exit')],
            ]

    return layout


    