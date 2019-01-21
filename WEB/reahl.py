class MyPage(HTML5Page):
  def __init__(self, view):
      super(MyPage, self).__init__(view)
      paragraph = P(view, text='Hello')
      self.body.add_child(paragraph)
