import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = 'browser'  # Ensure it opens in browser

t = np.linspace(0, 10, 100)
x = np.sin(t)
y = np.cos(t)
z = t

print (z)

fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='lines')])
fig.update_layout(title='3D Spiral')
fig.show()

