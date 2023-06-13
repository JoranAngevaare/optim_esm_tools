import matplotlib.pyplot as plt


def setup_map():
    import cartopy.crs as ccrs

    plt.gcf().add_subplot(
        projection=ccrs.PlateCarree(
            central_longitude=0.0,
        )
    )
    ax = plt.gca()
    ax.coastlines()
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
