import plotly.express as px

def show_3d_plot(points):
    # visualising
    fig = px.scatter_3d(
            x=points[:, 0],
            y=points[:, 2],
            z=points[:, 1], width = 720)

    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.update_traces(marker_size = 1)
    fig.show()

def show_3d_plot_shapenet(points):
    # visualising
    fig = px.scatter_3d(
            points, 
            x='x',
            y='z',
            z='y',
            color = 'label',             
            width = 720)

    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.update_traces(marker_size = 1)
    fig.show()